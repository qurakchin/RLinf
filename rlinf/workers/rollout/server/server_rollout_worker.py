# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import dataclasses
import json
import os
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from aiohttp import web, ClientSession
import aiohttp
import torch
from omegaconf import DictConfig
from transformers import AutoTokenizer

from rlinf.config import torch_dtype_from_precision
from rlinf.data.io_struct import (
    CompletionInfo,
    RolloutRequest,
    RolloutResult,
)
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.placement import ComponentPlacement
from rlinf.workers.rollout.sglang import Engine, io_struct
from rlinf.workers.rollout.utils import (
    print_sglang_outputs,
)
from toolkits.math_verifier.verify import MathRewardModel, math_verify_call


class TrainingDataStorage:
    """Storage manager for training data received via HTTP API."""
    
    def __init__(self, storage_config: Optional[Dict[str, Any]] = None):
        """
        Initialize storage manager.
        
        Args:
            storage_config: Configuration dict with options:
                - enabled: bool, whether to enable storage (default: True)
                - storage_dir: str, directory to store files (default: "./training_data")
                - file_format: str, format for files ("json" or "jsonl", default: "jsonl")
                - max_files_per_dir: int, max files per directory (default: 1000)
                - compress: bool, whether to compress files (default: False)
        """
        if storage_config is None:
            storage_config = {}
            
        self.enabled = storage_config.get('enabled', True)
        self.storage_dir = Path(storage_config.get('storage_dir', './training_data'))
        self.file_format = storage_config.get('file_format', 'jsonl')
        self.max_files_per_dir = storage_config.get('max_files_per_dir', 1000)
        self.compress = storage_config.get('compress', False)
        
        # Create storage directory if enabled
        if self.enabled:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            
        # Track current file and entry count
        self._current_file_path = None
        self._current_file_handle = None
        self._entries_in_current_file = 0
        
    async def store_training_data(self, training_data: Dict[str, Any]) -> Optional[str]:
        """
        Store training data to file.
        
        Args:
            training_data: The training data dictionary to store
            
        Returns:
            Path to the stored file, or None if storage is disabled
        """
        if not self.enabled:
            return None
            
        try:
            # Add metadata
            storage_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'stored_at': time.time(),
                'data': training_data
            }
            
            # Get or create file for writing
            file_path = await self._get_current_file_path()
            
            # Write data based on format
            if self.file_format == 'json':
                await self._write_json_entry(file_path, storage_entry)
            elif self.file_format == 'jsonl':
                await self._write_jsonl_entry(file_path, storage_entry)
            else:
                raise ValueError(f"Unsupported file format: {self.file_format}")
                
            return str(file_path)
            
        except Exception as e:
            # Log error but don't fail the main process
            print(f"Error storing training data: {str(e)}")
            return None
    
    async def _get_current_file_path(self) -> Path:
        """Get the current file path for writing, creating new file if needed."""
        # Check if we need a new file
        if (self._current_file_path is None or 
            self._entries_in_current_file >= self.max_files_per_dir):
            
            # Close current file if open
            if self._current_file_handle:
                self._current_file_handle.close()
                self._current_file_handle = None
            
            # Create new file path
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # microseconds to milliseconds
            filename = f"training_data_{timestamp}.{self.file_format}"
            if self.compress:
                filename += ".gz"
                
            self._current_file_path = self.storage_dir / filename
            self._entries_in_current_file = 0
            
        return self._current_file_path
    
    async def _write_json_entry(self, file_path: Path, entry: Dict[str, Any]):
        """Write entry to JSON file (array format)."""
        # For JSON format, we need to read existing data, append, and rewrite
        # This is less efficient but maintains valid JSON structure
        
        existing_data = []
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = []
        
        existing_data.append(entry)
        
        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)
            
        self._entries_in_current_file = len(existing_data)
    
    async def _write_jsonl_entry(self, file_path: Path, entry: Dict[str, Any]):
        """Write entry to JSONL file (one JSON per line)."""
        # JSONL is more efficient for appending
        with open(file_path, 'a', encoding='utf-8') as f:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
            
        self._entries_in_current_file += 1
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        if not self.enabled:
            return {"enabled": False}
            
        stats = {
            "enabled": True,
            "storage_dir": str(self.storage_dir),
            "file_format": self.file_format,
            "current_file": str(self._current_file_path) if self._current_file_path else None,
            "entries_in_current_file": self._entries_in_current_file,
            "total_files": 0,
            "total_size_bytes": 0
        }
        
        # Count files and calculate total size
        if self.storage_dir.exists():
            for file_path in self.storage_dir.iterdir():
                if file_path.is_file():
                    stats["total_files"] += 1
                    stats["total_size_bytes"] += file_path.stat().st_size
                    
        return stats
    
    async def cleanup(self):
        """Cleanup resources."""
        if self._current_file_handle:
            self._current_file_handle.close()
            self._current_file_handle = None


class UnifiedDataSource:
    """Unified data source that can handle both HTTP requests and Channel data."""
    
    def __init__(self, max_queue_size: int = 1000):
        self._queue = asyncio.Queue(maxsize=max_queue_size)
        self._http_enabled = True
        self._channel_enabled = True
        self._shutdown_event = asyncio.Event()
        
    async def put_http_data(self, training_data: Dict[str, Any]):
        """Put training data from HTTP request."""
        if self._http_enabled and not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(self._queue.put(('http', training_data)), timeout=5.0)
            except asyncio.TimeoutError:
                raise RuntimeError("Queue is full, cannot accept new HTTP data")
    
    async def put_channel_data(self, rollout_request: RolloutRequest):
        """Put rollout request from Channel."""
        if self._channel_enabled and not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(self._queue.put(('channel', rollout_request)), timeout=5.0)
            except asyncio.TimeoutError:
                raise RuntimeError("Queue is full, cannot accept new Channel data")
    
    async def get(self) -> Optional[Tuple[str, Union[Dict[str, Any], RolloutRequest]]]:
        """Get next data item regardless of source."""
        try:
            # Use wait_for with a timeout to allow periodic shutdown checks
            return await asyncio.wait_for(self._queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            if self._shutdown_event.is_set():
                return None
            raise
    
    def task_done(self):
        """Mark a task as done."""
        self._queue.task_done()
    
    def qsize(self) -> int:
        """Return the approximate size of the queue."""
        return self._queue.qsize()
    
    def enable_http(self, enabled: bool = True):
        """Enable/disable HTTP data source."""
        self._http_enabled = enabled
    
    def enable_channel(self, enabled: bool = True):
        """Enable/disable Channel data source."""
        self._channel_enabled = enabled
        
    async def shutdown(self):
        """Signal shutdown and drain remaining items."""
        self._shutdown_event.set()
        # Drain remaining items to unblock any waiting consumers
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break


class ServerRolloutWorker(Worker):
    """
    ServerRolloutWorker that supports both HTTP API and Channel interfaces.
    It can receive training data from router's feedback_worker via HTTP
    and also work with OnlineCodingRunner via Channel interface.
    
    Key features:
    - Unified data processing for both HTTP and Channel inputs
    - Automatic rollout processing after server startup
    - Compatible with OnlineCodingRunner interface
    """
    
    def __init__(self, config: DictConfig, placement: ComponentPlacement):
        Worker.__init__(self)

        self._cfg = config
        self._placement = placement

        # Initialize tokenizer for text processing
        self._tokenizer = AutoTokenizer.from_pretrained(self._cfg.rollout.model_dir)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        # Server configuration
        self._server_host = getattr(self._cfg.server, 'host', '0.0.0.0')
        self._server_port = getattr(self._cfg.server, 'port', 8081)
        
        # Unified data source for both HTTP and Channel data
        max_queue_size = getattr(self._cfg.server, 'max_queue_size', 1000)
        self._data_source = UnifiedDataSource(max_queue_size=max_queue_size)
        
        # Reward model for processing feedback
        self._reward_model = MathRewardModel(scale=self._cfg.reward.reward_scale)
        
        # Initialize training data storage
        storage_config = getattr(self._cfg, 'storage', None)
        if storage_config is not None:
            storage_config = dict(storage_config)
        self._storage = TrainingDataStorage(storage_config)
        
        # HTTP server components
        self._app = None
        self._runner = None
        self._site = None
        
        # Processing configuration
        self._batch_size = getattr(self._cfg.algorithm, 'rollout_batch_size_per_gpu', 1)
        self._max_new_tokens = getattr(self._cfg.algorithm.sampling_params, 'max_new_tokens', 512)
        
        # Processing control
        self._should_stop = False
        self._processing_tasks = []
        self._server_task = None
        
        # Event loop for async operations
        self._loop = None
        self._loop_thread = None
        
        # Output channel for continuous processing
        self._output_channel = None
        self._auto_processing_enabled = False

    def _start_event_loop(self):
        """Start event loop in a separate thread."""
        def run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()
        
        self._loop_thread = threading.Thread(target=run_loop, daemon=True)
        self._loop_thread.start()
        
        # Wait for loop to be ready
        while self._loop is None:
            time.sleep(0.01)

    def _stop_event_loop(self):
        """Stop the event loop and thread."""
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=5.0)

    async def _init_server(self):
        """Initialize the HTTP server to receive training data from router."""
        self.log_info(f"Initializing ServerRolloutWorker server on {self._server_host}:{self._server_port}")
        
        # Create aiohttp web application
        self._app = web.Application()
        
        # Add routes
        self._app.router.add_post('/api/training/submit', self._handle_training_data)
        self._app.router.add_get('/api/training/status/{job_id}', self._handle_status_query)
        self._app.router.add_get('/api/storage/stats', self._handle_storage_stats)
        self._app.router.add_get('/health', self._handle_health_check)
        
        # Create and start server
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        
        self._site = web.TCPSite(
            self._runner, 
            self._server_host, 
            self._server_port
        )
        await self._site.start()
        
        self.log_info(f"ServerRolloutWorker server started on http://{self._server_host}:{self._server_port}")

    async def _handle_health_check(self, request):
        """Handle health check requests."""
        return web.json_response({
            "status": "healthy",
            "timestamp": time.time(),
            "queue_size": self._data_source.qsize(),
            "processing": not self._should_stop,
            "auto_processing_enabled": self._auto_processing_enabled,
            "storage": self._storage.get_storage_stats()
        })

    async def _handle_training_data(self, request):
        """Handle training data submission from router's feedback_worker."""
        try:
            # Parse incoming training data
            training_data = await request.json()
            
            self.log_debug(f"Received training data: {training_data.get('metadata', {}).get('request_id', 'unknown')}")
            
            # Generate job ID
            job_id = f"job_{int(time.time() * 1000)}"
            
            # Add job metadata
            training_data['job_id'] = job_id
            training_data['received_at'] = time.time()
            
            # Store training data to file (async, non-blocking)
            storage_path = await self._storage.store_training_data(training_data)
            if storage_path:
                training_data['storage_path'] = storage_path
                self.log_debug(f"Training data stored to: {storage_path}")
            
            # Put data into unified data source
            await self._data_source.put_http_data(training_data)
            
            # Return response to router
            response_data = {
                "job_id": job_id,
                "status": "submitted",
                "message": "Training data submitted successfully",
                "queue_position": self._data_source.qsize()
            }
            
            return web.json_response(response_data)
            
        except Exception as e:
            self.log_error(f"Error handling training data: {str(e)}")
            return web.json_response(
                {"error": f"Failed to process training data: {str(e)}"}, 
                status=500
            )

    async def _handle_status_query(self, request):
        """Handle training job status queries."""
        job_id = request.match_info['job_id']
        
        return web.json_response({
            "job_id": job_id,
            "status": "processing",
            "message": "Job is being processed",
            "queue_size": self._data_source.qsize()
        })

    async def _handle_storage_stats(self, request):
        """Handle storage statistics requests."""
        try:
            stats = self._storage.get_storage_stats()
            return web.json_response({
                "status": "success",
                "timestamp": time.time(),
                "storage_stats": stats
            })
        except Exception as e:
            self.log_error(f"Error getting storage stats: {str(e)}")
            return web.json_response(
                {"error": f"Failed to get storage stats: {str(e)}"}, 
                status=500
            )

    def _convert_training_data_to_rollout_result(self, training_data: Dict[str, Any]) -> RolloutResult:
        """Convert training data from HTTP request into RolloutResult format."""
        try:
            # Extract text data
            input_text = training_data.get('input_text', '')
            output_text = training_data.get('output_text', '')
            reward_score = training_data.get('reward_score', 0.0)

            # Tokenize texts
            input_encoding = self._tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=self._cfg.runner.seq_length - self._max_new_tokens
            )
            input_ids = input_encoding['input_ids'][0].tolist()
            
            output_encoding = self._tokenizer(
                output_text,
                return_tensors="pt",
                truncation=True,
                max_length=self._max_new_tokens
            )
            output_ids = output_encoding['input_ids'][0].tolist()
            
            # Create RolloutResult with the feedback data
            group_size = getattr(self._cfg.algorithm, 'group_size', 1)
            
            rollout_result = RolloutResult(
                num_sequence=group_size,
                group_size=group_size,
                prompt_lengths=[len(input_ids)],
                prompt_ids=[input_ids],
                response_lengths=[len(output_ids)],
                response_ids=[output_ids],
                is_end=[True],  # Assume the response is complete
                rewards=torch.tensor([reward_score], dtype=torch.float32).reshape(-1, 1),
                advantages=[0.0],  # Will be computed later in the training pipeline
                prompt_texts=[input_text],
                response_texts=[output_text],
                answers=[output_text]
            )
            
            self.log_debug(f"Created RolloutResult from HTTP data with reward {reward_score}")
            
            return rollout_result
            
        except Exception as e:
            self.log_error(f"Error converting training data to RolloutResult: {str(e)}")
            raise

    def _convert_rollout_request_to_result(self, rollout_request: RolloutRequest) -> RolloutResult:
        """Convert RolloutRequest from Channel into RolloutResult format."""
        try:
            # For Channel data, we need to generate responses
            # This is a simplified implementation - in practice you might want to use SGLang/VLLM
            
            input_ids_list = rollout_request.input_ids
            answers = rollout_request.answers or []
            
            # Create mock responses for demonstration
            # In real implementation, you'd generate actual responses
            response_ids_list = []
            response_texts = []
            rewards = []
            
            for i, input_ids in enumerate(input_ids_list):
                # Mock response generation
                response_ids = input_ids[-10:] if len(input_ids) > 10 else input_ids
                response_ids_list.append(response_ids)
                
                response_text = self._tokenizer.decode(response_ids, skip_special_tokens=True)
                response_texts.append(response_text)
                
                # Mock reward calculation
                answer = answers[i] if i < len(answers) else ""
                reward = 1.0 if answer and answer in response_text else -1.0
                rewards.append(reward)
            
            rollout_result = RolloutResult(
                num_sequence=len(input_ids_list),
                group_size=rollout_request.n,
                prompt_lengths=[len(ids) for ids in input_ids_list],
                prompt_ids=input_ids_list,
                response_lengths=[len(ids) for ids in response_ids_list],
                response_ids=response_ids_list,
                is_end=[True] * len(input_ids_list),
                rewards=torch.tensor(rewards, dtype=torch.float32).reshape(-1, 1),
                advantages=[0.0] * len(input_ids_list),
                prompt_texts=[self._tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids_list],
                response_texts=response_texts,
                answers=answers
            )
            
            self.log_debug(f"Created RolloutResult from Channel data with {len(input_ids_list)} sequences")
            
            return rollout_result
            
        except Exception as e:
            self.log_error(f"Error converting RolloutRequest to RolloutResult: {str(e)}")
            raise

    async def _process_unified_data_continuously(self):
        """Continuously process data from the unified data source."""
        self.log_info("Starting continuous unified data processing")
        
        while not self._should_stop:
            try:
                # Get data from unified source (either HTTP or Channel)
                data_item = await self._data_source.get()
                
                if data_item is None:  # Shutdown signal
                    break
                
                data_type, data = data_item
                self.log_debug(f"Processing {data_type} data")
                
                # Convert data to RolloutResult based on source type
                if data_type == 'http':
                    rollout_result = self._convert_training_data_to_rollout_result(data)
                    job_id = data.get('job_id', 'unknown')
                    self.log_info(f"Processed HTTP training data: job_id={job_id}")
                elif data_type == 'channel':
                    rollout_result = self._convert_rollout_request_to_result(data)
                    self.log_info(f"Processed Channel rollout request with {len(data.input_ids)} sequences")
                else:
                    self.log_error(f"Unknown data type: {data_type}")
                    continue
                
                # Send result to output channel if available
                if self._output_channel:
                    self.log_info(f"Sending rollout result to output channel: job_id={job_id}")
                    await self._output_channel.put(item=rollout_result, async_op=True).async_wait()
                    # log the qsize of the output channel
                    self.log_info(f"Output channel qsize: {self._output_channel.qsize()}")
                
                # Mark task as done
                self._data_source.task_done()
                
            except asyncio.TimeoutError:
                # Normal timeout, continue processing
                continue
            except Exception as e:
                self.log_error(f"Error in continuous data processing: {str(e)}")
                # Mark task as done even if there was an error
                try:
                    self._data_source.task_done()
                except:
                    pass
        
        self.log_info("Continuous unified data processing stopped")

    async def _start_auto_processing(self, output_channel: Channel):
        """Start automatic data processing with the given output channel."""
        self._output_channel = output_channel
        self._auto_processing_enabled = True
        
        # Start continuous processing task
        processing_task = asyncio.create_task(self._process_unified_data_continuously())
        self._processing_tasks.append(processing_task)
        
        self.log_info("Auto processing started")

    async def _run_server_with_auto_processing(self, output_channel: Channel):
        """Run HTTP server and start automatic data processing."""
        try:
            # Start HTTP server
            await self._init_server()
            
            # Start automatic processing
            await self._start_auto_processing(output_channel)
            
            self.log_info("ServerRolloutWorker is running with HTTP server and auto processing")
            
            # Keep server running until shutdown
            while not self._should_stop:
                await asyncio.sleep(1.0)
            
        except Exception as e:
            self.log_error(f"Error in server with auto processing: {str(e)}")
            raise

    def init_worker(self):
        """Initialize the worker (sync version)."""
        self.log_info("Initializing ServerRolloutWorker")
        
        # Start event loop in separate thread
        self._start_event_loop()
        
        self.log_info("ServerRolloutWorker initialized")

    def start_server_with_auto_processing(self, output_channel: Channel):
        """Start HTTP server and automatic data processing (sync interface)."""
        if self._server_task is not None:
            self.log_warning("Server is already running")
            return
        
        # Schedule the async operation
        future = asyncio.run_coroutine_threadsafe(
            self._run_server_with_auto_processing(output_channel), 
            self._loop
        )
        
        # Store the future for later cleanup, but don't return Handle
        self._server_task = future

    def rollout(self, input_channel: Optional[Channel] = None, output_channel: Optional[Channel] = None):
        """
        Main rollout method that supports both Channel interface (for OnlineCodingRunner)
        and standalone HTTP server mode.
        
        Like SGLang Worker, this method doesn't return Handle - it just starts the processing
        and lets it run in the background.
        """
        
        if input_channel is not None and output_channel is not None:
            # Channel interface mode - integrate with OnlineCodingRunner
            self.log_info("Starting rollout in Channel interface mode")
            
            async def channel_rollout():
                try:
                    # Enable channel data source
                    self._data_source.enable_channel(True)
                    
                    # Get rollout request from input channel
                    rollout_request = await input_channel.get(async_op=True).async_wait()
                    
                    # Put request into unified data source
                    await self._data_source.put_channel_data(rollout_request)
                    
                    # If auto processing is not enabled, start it
                    if not self._auto_processing_enabled:
                        await self._start_auto_processing(output_channel)
                    
                    # If server is not running, start it
                    if self._server_task is None:
                        server_task = asyncio.create_task(
                            self._run_server_with_auto_processing(output_channel)
                        )
                        self._processing_tasks.append(server_task)
                    
                    # The processing will be handled automatically by the continuous processor
                    
                except Exception as e:
                    self.log_error(f"Error in channel rollout: {str(e)}")
                    raise
            
            # Schedule the async operation (no return, like SGLang worker)
            asyncio.run_coroutine_threadsafe(channel_rollout(), self._loop)
            
        elif output_channel is not None:
            # Standalone HTTP server mode with specified output channel  
            self.log_info("Starting rollout in standalone HTTP server mode with output channel")
            
            # Start server (no return, just start the service)
            self.start_server_with_auto_processing(output_channel)
            
        else:
            # Pure standalone HTTP server mode
            self.log_info("Starting rollout in pure standalone HTTP server mode")
            
            # Create a dummy output channel for standalone mode
            standalone_output_channel = Channel.create("StandaloneOutput", local=True)
            self.start_server_with_auto_processing(standalone_output_channel)
        
        # Like SGLang worker, return nothing (None)

    async def shutdown(self):
        """Shutdown the server and cleanup resources."""
        self.log_info("Shutting down ServerRolloutWorker")
        
        # Set stop flag
        self._should_stop = True
        
        # Shutdown data source
        await self._data_source.shutdown()
        
        # Stop HTTP server
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()
        
        # Cancel all processing tasks
        for task in self._processing_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._processing_tasks:
            await asyncio.gather(*self._processing_tasks, return_exceptions=True)
        
        # Cleanup storage
        await self._storage.cleanup()
        
        self.log_info("ServerRolloutWorker shutdown complete")

    def _stop(self):
        """Stop the server and cleanup resources (sync version)."""
        self.log_info("Stopping ServerRolloutWorker...")
        
        self._should_stop = True
        
        # Schedule shutdown in event loop
        if self._loop and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(self.shutdown(), self._loop)
        
        # Stop event loop
        self._stop_event_loop()