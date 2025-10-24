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
import logging
import time
import uuid
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Any, List
from enum import Enum

from omegaconf import DictConfig

from rlinf.scheduler import Worker, Channel

# MCP imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from rlinf.data.MCP_io_struct import MCPRequest, MCPRequestType, MCPResponse, MCPSessionState
from rlinf.workers.mcp.tool_worker import ToolWorker
from rlinf.data.tool_call.tool_io_struct import ToolResponse


class PythonSandboxSession:
    """MCP session manager for Python sandbox execution."""
    
    def __init__(self, session_id: str, config: DictConfig):
        self.session_id = session_id
        self.config = config
        self.state = MCPSessionState.INITIALIZING
        
        # Session resources
        self.server_params: Optional[StdioServerParameters] = None
        
        # Statistics
        self.created_at = time.time()
        self.last_activity = time.time()
        self.requests_processed = 0
        
        # Execution history (optional, for debugging)
        self.execution_history: List[Dict[str, Any]] = []
        self.max_history_size = config.get('mcp', {}).get('max_history_size', 100)
        
        self._logger = None
    
    @property
    def logger(self):
        """Lazy initialization of logger."""
        if self._logger is None:
            self._logger = logging.getLogger(f"PythonSandboxSession.{self.session_id}")
        return self._logger
    
    async def start(self) -> bool:
        """Start the MCP session."""
        try:
            self.logger.info(f"Starting Python sandbox session {self.session_id}")
            
            # Create server parameters for deno stdio connection
            self.server_params = StdioServerParameters(
                command="deno",
                args=[
                    "run",
                    "-N",
                    "-R=node_modules",
                    "-W=node_modules",
                    "--node-modules-dir=auto",
                    "jsr:@pydantic/mcp-run-python",
                    "stdio"
                ]
            )
            async with stdio_client(self.server_params) as (read, write):
                # print(f"Starting Python sandbox session {self.session_id} after stdio_client")
                async with ClientSession(read, write) as session:
                    print(f"Starting Python sandbox session {self.session_id} after ClientSession")
                    await session.initialize()
                    print(f"Starting Python sandbox session {self.session_id} after initialize")
                    # Optionally list tools to verify connection
                    tools = await session.list_tools()
                    self.logger.info(f"Available tools: {[tool.name for tool in tools.tools]}")
            
            self.state = MCPSessionState.CONNECTED
            self.logger.info(f"Python sandbox session {self.session_id} started successfully")
            print(f"Python sandbox session {self.session_id} started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting Python sandbox session {self.session_id}: {e}")
            self.state = MCPSessionState.FAILED
            return False
    
    async def process_request(self, request: MCPRequest) -> MCPResponse:
        """Process an MCP request."""
        start_time = time.time()
        self.last_activity = start_time
        
        try:
            if self.state != MCPSessionState.CONNECTED:
                return MCPResponse(
                    request_id=request.request_id,
                    success=False,
                    error_message="Python sandbox session not connected",
                    execution_time=time.time() - start_time
                )
            
            self.logger.debug(f"Processing Python sandbox request: {request.request_type}")
            
            # Open stdio and client session per request
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    # Process based on request type
                    if request.request_type == MCPRequestType.LIST_TOOLS:
                        result = await session.list_tools()
                        result_data = {
                            "tools": [
                                {
                                    "name": tool.name,
                                    "description": tool.description,
                                    "input_schema": tool.inputSchema if hasattr(tool, 'inputSchema') else None
                                }
                                for tool in result.tools
                            ]
                        }
                    
                    elif request.request_type == MCPRequestType.CALL_TOOL:
                        if not request.tool_name:
                            raise ValueError("Tool name is required for tool calls")
                        
                        # For Python sandbox, the main tool is "run_python_code"
                        if request.tool_name != "run_python_code":
                            raise ValueError(f"Unknown tool: {request.tool_name}. Only 'run_python_code' is supported.")
                        
                        # Get Python code from arguments
                        # Note: The tool expects 'python_code' not 'code'
                        code = request.tool_arguments.get('code') or request.tool_arguments.get('python_code', '')
                        if not code:
                            raise ValueError("Python code is required in tool_arguments['code'] or tool_arguments['python_code']")
                        
                        # Prepare arguments with correct parameter name
                        tool_args = {"python_code": code}
                        
                        self.logger.info(f"[DEBUG] Calling MCP tool with code length: {len(code)}")
                        self.logger.info(f"[DEBUG] Tool args: {tool_args}")
                        
                        # Execute Python code
                        result = await session.call_tool(
                            request.tool_name,
                            tool_args
                        )
                        
                        self.logger.info(f"[DEBUG] MCP call_tool returned, result type: {type(result)}")
                        self.logger.info(f"[DEBUG] Result attributes: {dir(result)}")
                        self.logger.info(f"[DEBUG] Result.content: {result.content if hasattr(result, 'content') else 'NO CONTENT ATTR'}")
                        
                        # Extract content
                        content = []
                        for item in result.content:
                            self.logger.info(f"[DEBUG] Processing content item type: {type(item)}, has text: {hasattr(item, 'text')}")
                            if hasattr(item, 'text'):
                                content.append(item.text)
                            else:
                                content.append(str(item))
                        
                        self.logger.info(f"[DEBUG] Extracted content: {content}")
                        
                        result_data = {
                            "content": content,
                            "output": '\n'.join(content),
                            "is_error": result.isError if hasattr(result, 'isError') else False
                        }
                        
                        self.logger.info(f"[DEBUG] Final result_data: {result_data}")
                        
                        # Store execution history
                        self._add_to_history({
                            "timestamp": time.time(),
                            "code": code,
                            "output": result_data["output"],
                            "is_error": result_data["is_error"]
                        })
                    
                    elif request.request_type == MCPRequestType.LIST_RESOURCES:
                        result = await session.list_resources()
                        result_data = {
                            "resources": [
                                {"uri": resource.uri, "name": resource.name}
                                for resource in result.resources
                            ]
                        }
                    
                    elif request.request_type == MCPRequestType.READ_RESOURCE:
                        if not request.resource_uri:
                            raise ValueError("Resource URI is required for resource reading")
                        from pydantic import AnyUrl
                        result = await session.read_resource(AnyUrl(request.resource_uri))
                        content = []
                        for item in result.contents:
                            if hasattr(item, 'text'):
                                content.append(item.text)
                            else:
                                content.append(str(item))
                        result_data = {"content": content}
                    
                    elif request.request_type == MCPRequestType.LIST_PROMPTS:
                        result = await session.list_prompts()
                        result_data = {
                            "prompts": [
                                {"name": prompt.name, "description": prompt.description}
                                for prompt in result.prompts
                            ]
                        }
                    
                    elif request.request_type == MCPRequestType.GET_PROMPT:
                        if not request.prompt_name:
                            raise ValueError("Prompt name is required for prompt getting")
                        result = await session.get_prompt(
                            request.prompt_name,
                            request.prompt_arguments or {}
                        )
                        messages = []
                        for msg in result.messages:
                            message_data = {
                                "role": msg.role,
                                "content": []
                            }
                            for content_item in msg.content:
                                if hasattr(content_item, 'text'):
                                    message_data["content"].append({
                                        "type": "text",
                                        "text": content_item.text
                                    })
                                else:
                                    message_data["content"].append({
                                        "type": str(type(content_item).__name__),
                                        "data": str(content_item)
                                    })
                            messages.append(message_data)
                        result_data = {"messages": messages}
                    
                    else:
                        raise ValueError(f"Unknown request type: {request.request_type}")
            
            self.requests_processed += 1
            execution_time = time.time() - start_time
            
            return MCPResponse(
                request_id=request.request_id,
                success=True,
                result=result_data,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            import traceback
            error_details = traceback.format_exc()
            
            # 如果是 ExceptionGroup，提取所有子异常
            error_message = str(e)
            if hasattr(e, 'exceptions'):  # ExceptionGroup
                sub_errors = []
                for sub_e in e.exceptions:
                    sub_errors.append(f"  - {type(sub_e).__name__}: {str(sub_e)}")
                error_message = f"{error_message}\n子异常:\n" + "\n".join(sub_errors)
            
            self.logger.error(f"Error processing Python sandbox request {request.request_id}: {e}\n{error_details}")
            
            return MCPResponse(
                request_id=request.request_id,
                success=False,
                error_message=f"{error_message}\n\n详细堆栈:\n{error_details}",
                execution_time=execution_time
            )
    
    def _add_to_history(self, entry: Dict[str, Any]):
        """Add execution to history with size limit."""
        self.execution_history.append(entry)
        if len(self.execution_history) > self.max_history_size:
            self.execution_history.pop(0)
    
    async def cleanup(self):
        """Cleanup session resources."""
        try:
            self.state = MCPSessionState.TERMINATED
            self.logger.info(f"Python sandbox session {self.session_id} cleaned up")
        except Exception as e:
            self.logger.error(f"Error cleaning up Python sandbox session {self.session_id}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "requests_processed": self.requests_processed,
            "uptime": time.time() - self.created_at,
            "history_size": len(self.execution_history)
        }
    
    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get execution history."""
        if limit:
            return self.execution_history[-limit:]
        return self.execution_history.copy()


class MCPPythonSandboxWorker(ToolWorker):
    """MCP Python Sandbox Worker for executing Python code in isolated environment."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.is_running = False
        
        # Session management
        self.sessions: Dict[str, PythonSandboxSession] = {}
        self._session_lock = None
        
        # Configuration
        self.default_timeout = cfg.get('mcp', {}).get('default_timeout', 30)

    @property
    def session_lock(self):
        """Lazy initialization of session lock."""
        if self._session_lock is None:
            self._session_lock = threading.RLock()
        return self._session_lock

    @property
    def logger(self):
        """Lazy initialization of logger."""
        if self._logger is None:
            self._logger = logging.getLogger("MCPPythonSandboxWorker")
        return self._logger

    def init_worker(self, input_channel: Channel, output_channel: Channel):
        """Initialize the worker with communication channels."""
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.is_running = True
        
        # Initialize components
        self.sessions = {}
        
        self.log_info("MCP Python Sandbox Worker initialized")
        return self

    def start_server(self):
        """Start the request processor task."""
        try:
            loop = asyncio.get_running_loop()
            self.request_processor_task = loop.create_task(self._process_requests())
        except RuntimeError:
            # No event loop running, will be created later
            self.request_processor_task = None

    async def _process_requests(self):
        """Process incoming requests asynchronously."""
        self.log_info("Starting async request processor")
        
        while self.is_running:
            try:
                # Get request from input channel
                self.logger.info("[DEBUG] Waiting for request from input_channel...")
                rollout_request = await self.input_channel.get(async_op=True).async_wait()
                channel_key, tool_args = rollout_request['channel_key'], rollout_request['tool_args']
                request_work = MCPRequest(
                    request_id=str(uuid.uuid4()),
                    request_type=MCPRequestType.CALL_TOOL,
                    tool_name="run_python_code",
                    tool_arguments=tool_args,
                    timeout=self.default_timeout,
                    metadata={"session_id": channel_key},
                )
                # self.logger.info(f"[DEBUG] Got request: {request_work.request_type if hasattr(request_work, 'request_type') else type(request_work)}")
                
                # Process request
                result = await self._handle_request(request_work)
                self.logger.info(f"[DEBUG] Request processed, result success: {result.success if hasattr(result, 'success') else 'N/A'}")
                self.logger.info(f"[DEBUG] Request result is: {result}")
                
                # Send result to output channel
                # self.output_channel.put(result, async_op=True)
                result_dict = ToolResponse(text=str(result.result))

                self.output_channel.put(result_dict, key=channel_key, async_op=True)
                self.logger.info("[DEBUG] Result sent to output_channel")

            except Exception as e:
                if "QueueEmpty" not in str(e):
                    self.logger.error(f"Error processing request: {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                await asyncio.sleep(0.1)
    
    async def _handle_request(self, request: Any) -> MCPResponse:
        """Handle different types of requests."""
        try:
            if isinstance(request, MCPRequest):
                return await self._process_mcp_request(request)
            elif isinstance(request, dict):
                mcp_request = self._dict_to_mcp_request(request)
                return await self._process_mcp_request(mcp_request)
            else:
                return MCPResponse(
                    request_id=str(uuid.uuid4()),
                    success=False,
                    error_message=f"Unsupported request type: {type(request)}"
                )
        except Exception as e:
            self.logger.error(f"Error handling request: {e}")
            return MCPResponse(
                request_id=str(uuid.uuid4()),
                success=False,
                error_message=str(e)
            )
    
    def _dict_to_mcp_request(self, data: dict) -> MCPRequest:
        """Convert dictionary to MCPRequest."""
        return MCPRequest(
            request_id=data.get('request_id', str(uuid.uuid4())),
            request_type=MCPRequestType(data.get('request_type', 'call_tool')),
            tool_name=data.get('tool_name'),
            tool_arguments=data.get('tool_arguments', {}),
            resource_uri=data.get('resource_uri'),
            prompt_name=data.get('prompt_name'),
            prompt_arguments=data.get('prompt_arguments', {}),
            timeout=data.get('timeout', 30),
            metadata=data.get('metadata', {})
        )
    
    async def _process_mcp_request(self, request: MCPRequest) -> MCPResponse:
        """Process a single MCP request."""
        session_id = request.metadata.get('session_id', 'default')
        
        try:
            # Get or create session
            session = await self._get_or_create_session(session_id)
            
            if not session:
                return MCPResponse(
                    request_id=request.request_id,
                    success=False,
                    error_message="Failed to create Python sandbox session"
                )
            
            # Use request.timeout directly, fallback to default_timeout if not set
            timeout = request.timeout if request.timeout else self.default_timeout
            
            try:
                response = await asyncio.wait_for(
                    session.process_request(request),
                    timeout=timeout
                )
                return response
            except asyncio.TimeoutError:
                return MCPResponse(
                    request_id=request.request_id,
                    success=False,
                    error_message=f"Request timed out after {timeout} seconds"
                )
        
        except Exception as e:
            self.logger.error(f"Error processing MCP request: {e}")
            return MCPResponse(
                request_id=request.request_id,
                success=False,
                error_message=str(e)
            )
    
    async def _get_or_create_session(self, session_id: str) -> Optional[PythonSandboxSession]:
        """Get existing session or create new one."""
        print(f"session_id: {session_id}, sessions: {self.sessions}")
        async with asyncio.Lock():
            if session_id in self.sessions:
                session = self.sessions[session_id]
                if session.state == MCPSessionState.CONNECTED:
                    return session
                else:
                    # Remove failed/terminated session
                    await self._remove_session(session_id)
            
            # Create new session
            session = PythonSandboxSession(session_id, self.cfg)
            print(f"session: {session}")
            if await session.start():
                self.sessions[session_id] = session
                self.log_info(f"Created new Python sandbox session: {session_id}")
                return session
            else:
                self.logger.error(f"Failed to create Python sandbox session: {session_id}")
                return None
    
    async def _remove_session(self, session_id: str):
        """Remove session from manager."""
        if session_id in self.sessions:
            session = self.sessions.pop(session_id)
            await session.cleanup()
            self.log_info(f"Removed Python sandbox session: {session_id}")
    
    async def _cleanup_all_sessions(self):
        """Cleanup all sessions."""
        tasks = []
        for session_id in list(self.sessions.keys()):
            tasks.append(self._remove_session(session_id))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def cleanup(self):
        """Cleanup worker resources."""
        if self.logger:
            self.log_info("Cleaning up MCP Python Sandbox Worker")
        
        self.is_running = False
        
        # Cancel request processor task
        if hasattr(self, 'request_processor_task') and self.request_processor_task and not self.request_processor_task.done():
            self.request_processor_task.cancel()
        
        # Cleanup all sessions
        if self.sessions:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._cleanup_all_sessions())
            except RuntimeError:
                pass
        
        return self
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        session_stats = []
        for session in self.sessions.values():
            session_stats.append(session.get_stats())
        
        return {
            "worker_type": "MCPPythonSandboxWorker",
            "is_running": self.is_running,
            "sessions": session_stats,
            "total_sessions": len(self.sessions),
            "default_timeout": self.default_timeout
        }
    
    def get_session_history(self, session_id: str, limit: Optional[int] = None) -> Optional[List[Dict[str, Any]]]:
        """Get execution history for a specific session."""
        if session_id in self.sessions:
            return self.sessions[session_id].get_history(limit)
        return None
    
    def __getstate__(self):
        """Custom serialization to exclude non-serializable objects."""
        state = self.__dict__.copy()
        serializable_keys = ['cfg', 'is_running', 'default_timeout']
        return {k: v for k, v in state.items() if k in serializable_keys}
    
    def __setstate__(self, state):
        """Custom deserialization to restore non-serializable objects."""
        self.__dict__.update(state)
        self.sessions = {}
        self._session_lock = None
        self.request_processor_task = None
        self.input_channel = None
        self.output_channel = None


# Convenience function for quick Python code execution
async def execute_python_code(
    code: str,
    session_id: str = "default",
    timeout: int = 30,
    worker: Optional[MCPPythonSandboxWorker] = None
) -> MCPResponse:
    """
    Convenience function to execute Python code.
    
    Args:
        code: Python code to execute
        session_id: Session identifier
        timeout: Execution timeout in seconds
        worker: Optional worker instance (creates new one if not provided)
    
    Returns:
        MCPResponse with execution results
    """
    request = MCPRequest(
        request_id=str(uuid.uuid4()),
        request_type=MCPRequestType.CALL_TOOL,
        tool_name="run_python_code",
        tool_arguments={"code": code},
        timeout=timeout,
        metadata={"session_id": session_id}
    )
    
    if worker:
        return await worker._process_mcp_request(request)
    else:
        # Create temporary worker
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({"mcp": {"default_timeout": timeout}})
        temp_worker = MCPPythonSandboxWorker(cfg)
        
        # Create temporary session
        session = PythonSandboxSession(session_id, cfg)
        await session.start()
        
        try:
            return await session.process_request(request)
        finally:
            await session.cleanup()
