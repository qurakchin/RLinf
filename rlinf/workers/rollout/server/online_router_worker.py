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
import json
import logging
import random
import time
import uuid
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from omegaconf.dictconfig import DictConfig
from pydantic import BaseModel
import uvicorn

from rlinf.scheduler import Channel, Worker
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.workers.rollout.sglang.sglang_worker import AsyncSGLangWorker
from rlinf.utils.placement import ComponentPlacement

logger = logging.getLogger(__name__)


class CompleteRequest(BaseModel):
    """Complete request model."""
    prompt: str
    model: Optional[str] = None
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    repetition_penalty: Optional[float] = 1.0
    stop: Optional[List[str]] = None
    stream: Optional[bool] = False

class CompletionTrackRequest(BaseModel):
    """Completion track request model."""
    data: Dict[str, Any]
    batch_size: Optional[int] = None


class CompleteResponse(BaseModel):
    """Complete response model."""
    # request_id: str
    # response_id: str
    # generated_text: str
    # tokens_used: Optional[int] = None
    # latency_ms: float
    # status: str = "success"
    id: str
    choices: List[Dict[str, Any]]
    model: str
    created: int
    object: str='text_completion'


class OnlineRouterWorker(Worker):
    """Online router worker with FastAPI server for handling complete and completionTrack requests."""

    def __init__(self, cfg: DictConfig, placement: ComponentPlacement):
        Worker.__init__(self)
        self.log_info("zcy_dbg: OnlineRouterWorker:init: 1")

        self.cfg = cfg
        self.app = FastAPI(title="OnlineRouterWorker", version="1.0.0")

        # Configuration
        # self.server_host = cfg.get('server_host', '0.0.0.0')
        # self.server_port = cfg.get('server_port', 8081)
        self.server_host = '0.0.0.0'
        self.server_port = 8081
        self.rollout_instance_num = placement.rollout_dp_size

        # Sync weight state management
        self._sync_weight_lock = asyncio.Lock()
        self._sync_weight_in_progress = False
        self._pending_requests: List[asyncio.Future] = []
        
        # Request synchronization state
        self._sync_in_progress = False
        self._old_requests_complete = asyncio.Event()
        self._new_requests_blocked = asyncio.Event()
        self._new_requests_blocked.set()  # Initially allow new requests
        self._blocked_requests: List[asyncio.Future] = []

        # Request tracking
        self._active_requests: Dict[str, asyncio.Future] = {}

        # Setup FastAPI routes
        self._setup_routes()
        self.log_info("zcy_dbg: OnlineRouterWorker:init: 2")

        self._server_running = False

    def _setup_routes(self):
        """Setup FastAPI routes."""

        @self.app.post("/v1/completions")
        async def complete(request: CompleteRequest):
            """Handle complete requests."""
            logger.info("zcy_dbg: OnlineRouterWorker:complete: 1")
            self.log_info("zcy_dbg: OnlineRouterWorker:complete: 1")
            self.log_info(f"zcy_dbg: OnlineRouterWorker:complete: 2: {request}")
            return await self._handle_complete(request)

        @self.app.get("/health")
        async def a():
            """Health check endpoint."""
            return {"status": "healthy", "timestamp": time.time()}

        @self.app.get("/status")
        async def status():
            """Status endpoint."""
            return {
                "status": "running",
                "sync_weight_in_progress": self._sync_weight_in_progress,
                "sync_in_progress": self._sync_in_progress,
                "active_requests": len(self._active_requests),
                "blocked_requests": len(self._blocked_requests),
                "timestamp": time.time()
            }

    async def _handle_complete(self, request: CompleteRequest) -> CompleteResponse:
        """Handle complete requests with synchronization support."""
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Check if sync is in progress
        if self._sync_in_progress:
            # Wait for old requests to complete
            await self._old_requests_complete.wait()
            # Block new requests during sync
            await self._new_requests_blocked.wait()
        
        # Create future for this request
        future = asyncio.Future()
        self._active_requests[request_id] = future

        try:
            # Forward request to rollout worker
            sglang_instance_id = random.randint(0, self.rollout_instance_num - 1)
            generate_result = await self.rollout_worker.execute_on(sglang_instance_id).agenerate(request.prompt).async_wait()
            generated_text = generate_result[0]['text']
            self.log_info(f"zcy_dbg: OnlineRouterWorker:complete: 4: generated_text=[{generated_text}]")

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            if not request.stream:
                # Create response
                response = CompleteResponse(
                    id=str(request_id),
                    choices=[{
                        "text": generated_text,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "stop"
                    }],
                    created=int(start_time),
                    model="test-model",
                    object="text_completion"
                )
            else:
                def generate_stream():
                    # Send final chunk with finish_reason
                    final_data = {
                        "id": request_id,
                        "object": "text_completion.chunk",
                        "created": int(start_time),
                        "model": "test-model",
                        "choices": [{
                            "text": generated_text,
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": "stop"
                        }]
                    }
                    yield f"data: {json.dumps(final_data)}\n\n"
                    yield "data: [DONE]\n\n"

                response = StreamingResponse(
                    generate_stream(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",  # Disable nginx buffering
                    }
                )
            
            # Set future result
            future.set_result(response)
            return response

        finally:
            # Clean up
            if request_id in self._active_requests:
                del self._active_requests[request_id]

    async def init_worker(self, rollout_worker: AsyncSGLangWorker):
        """Initialize the worker."""
        self.rollout_worker = rollout_worker

    def rollout_start(self):
        """Start rollout service."""
        assert not self._server_running
        self._server_running = True
        
        # Start the HTTP server
        config = uvicorn.Config(
            self.app,
            host=self.server_host,
            port=self.server_port,
            log_level="info"
        )
        self._server = uvicorn.Server(config)
        
        # Start server in background task
        asyncio.create_task(self._server.serve())

        logger.info(f"Rollout service started on {self.server_host}:{self.server_port}")

    def rollout_stop(self):
        """Stop rollout service."""
        assert self._server_running
        self._server_running = False
        
        # Stop the HTTP server
        if hasattr(self, '_server') and self._server:
            self._server.should_exit = True
        
        logger.info("Rollout service stopped")

    async def sync_model_start(self):
        """Start model synchronization. Block new requests and wait for old ones to complete."""
        async with self._sync_weight_lock:
            assert not self._sync_in_progress
            
            logger.info("Starting model synchronization...")
            self._sync_in_progress = True
            
            # Clear the event to block new requests
            self._new_requests_blocked.clear()
            
            # Wait for all existing requests to complete
            if self._active_requests:
                logger.info(f"Waiting for {len(self._active_requests)} active requests to complete...")
                # Wait for all active requests to finish
                await asyncio.gather(*self._active_requests.values(), return_exceptions=True)
            
            # Set event to indicate old requests are complete
            self._old_requests_complete.set()
            logger.info("All old requests completed, sync can proceed")

    async def sync_model_end(self):
        """End model synchronization. Resume processing of blocked requests."""
        async with self._sync_weight_lock:
            assert self._sync_in_progress
            
            logger.info("Ending model synchronization...")
            
            # Reset sync state
            self._sync_in_progress = False
            self._old_requests_complete.clear()
            
            # Allow new requests to proceed
            self._new_requests_blocked.set()
            
            logger.info("Model synchronization completed, new requests can proceed")
