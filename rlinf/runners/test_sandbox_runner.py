import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Callable

from omegaconf import DictConfig

from rlinf.scheduler import Channel, Worker
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.timers import Timer
from rlinf.workers.mcp.sandbox.mcp_sandbox_worker import MCPPythonSandboxWorker
from rlinf.data.MCP_io_struct import MCPRequest, MCPRequestType, MCPResponse

logging.getLogger().setLevel(logging.INFO)


class MCPTestRunner:
    """MCP Test Runner for Python Sandbox Worker."""

    def __init__(self, cfg: DictConfig, client_worker: MCPPythonSandboxWorker):
        self.cfg = cfg
        self.client_worker = client_worker

        # Communication channels
        self.input_channel = Channel.create("ClientInput", local=False)
        self.output_channel = Channel.create("ClientOutput", local=False)

    def init_workers(self):
        """Initialize the workers."""
        self.client_worker.init_worker(self.input_channel, self.output_channel)

        logging.info("MCP Python Sandbox Test Runner initialized")

    def cleanup(self):
        """Cleanup the workers."""
        self.client_worker.cleanup()

    def run(self):
        """Run the MCP Python sandbox test."""
        session_id1 = str(uuid.uuid4())
        session_id2 = str(uuid.uuid4())
        print(f"\n{'='*70}")
        print(f"Starting MCP Python Sandbox Tests")
        print(f"Session 1 ID: {session_id1}")
        print(f"Session 2 ID: {session_id2}")
        print(f"{'='*70}\n")
        
        # 适配 Python Sandbox Worker 的测试任务
        tasks = [
            # 1. 列出可用工具
            MCPRequest(
                request_id=str(uuid.uuid4()),
                request_type=MCPRequestType.LIST_TOOLS,
                tool_name=None,
                tool_arguments=None,
                resource_uri=None,
                prompt_name=None,
                prompt_arguments=None,
                timeout=30,
                metadata={
                    "session_id": session_id1,
                }
            ),
            
            # 2. 执行简单的 Python 代码
            MCPRequest(
                request_id=str(uuid.uuid4()),
                request_type=MCPRequestType.CALL_TOOL,
                tool_name="run_python_code",
                tool_arguments={
                    "python_code": "print('Hello, World!')"
                },
                resource_uri=None,
                prompt_name=None,
                prompt_arguments=None,
                timeout=30,
                metadata={
                    "session_id": session_id1
                }
            ),
            
            # 3. 执行数学计算
            MCPRequest(
                request_id=str(uuid.uuid4()),
                request_type=MCPRequestType.CALL_TOOL,
                tool_name="run_python_code",
                tool_arguments={
                    "python_code": """
import math

# 计算圆的面积
radius = 5
area = math.pi * radius ** 2
print(f"半径为 {radius} 的圆的面积是: {area:.2f}")

# 计算阶乘
n = 10
factorial = math.factorial(n)
print(f"{n}! = {factorial}")
"""
                },
                resource_uri=None,
                prompt_name=None,
                prompt_arguments=None,
                timeout=30,
                metadata={
                    "session_id": session_id1
                }
            ),
            
            # 4. 执行数据处理
            MCPRequest(
                request_id=str(uuid.uuid4()),
                request_type=MCPRequestType.CALL_TOOL,
                tool_name="run_python_code",
                tool_arguments={
                    "python_code": """
# 数据处理示例
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 计算统计信息
total = sum(data)
count = len(data)
average = total / count
maximum = max(data)
minimum = min(data)

print(f"数据: {data}")
print(f"总和: {total}")
print(f"平均值: {average}")
print(f"最大值: {maximum}")
print(f"最小值: {minimum}")

# 过滤偶数
evens = [x for x in data if x % 2 == 0]
print(f"偶数: {evens}")
"""
                },
                resource_uri=None,
                prompt_name=None,
                prompt_arguments=None,
                timeout=30,
                metadata={
                    "session_id": session_id1
                }
            ),
            
            # 5. 测试会话隔离 - 在另一个会话中执行代码
            MCPRequest(
                request_id=str(uuid.uuid4()),
                request_type=MCPRequestType.CALL_TOOL,
                tool_name="run_python_code",
                tool_arguments={
                    "python_code": """
print(f"这是会话 2 的代码执行")
session_var = "Session 2 Variable"
print(f"会话变量: {session_var}")
"""
                },
                resource_uri=None,
                prompt_name=None,
                prompt_arguments=None,
                timeout=30,
                metadata={
                    "session_id": session_id2
                }
            ),
            
            # 6. 测试时间戳和字符串操作
            MCPRequest(
                request_id=str(uuid.uuid4()),
                request_type=MCPRequestType.CALL_TOOL,
                tool_name="run_python_code",
                tool_arguments={
                    "python_code": f"""
import time

# 当前时间戳
timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
print(f"当前时间: {{timestamp}}")

# 字符串操作
text = "Python Sandbox Test"
print(f"原始文本: {{text}}")
print(f"大写: {{text.upper()}}")
print(f"小写: {{text.lower()}}")
print(f"单词数: {{len(text.split())}}")
"""
                },
                resource_uri=None,
                prompt_name=None,
                prompt_arguments=None,
                timeout=30,
                metadata={
                    "session_id": session_id1
                }
            ),
            
            # 7. 测试错误处理
            MCPRequest(
                request_id=str(uuid.uuid4()),
                request_type=MCPRequestType.CALL_TOOL,
                tool_name="run_python_code",
                tool_arguments={
                    "python_code": """
try:
    # 故意制造一个错误
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"捕获到错误: {e}")
    print("错误已被正确处理")

print("程序继续执行")
"""
                },
                resource_uri=None,
                prompt_name=None,
                prompt_arguments=None,
                timeout=30,
                metadata={
                    "session_id": session_id1
                }
            ),
            
            # 8. 测试 JSON 处理
            MCPRequest(
                request_id=str(uuid.uuid4()),
                request_type=MCPRequestType.CALL_TOOL,
                tool_name="run_python_code",
                tool_arguments={
                    "python_code": """
import json

# 创建 JSON 数据
data = {
    "name": "Python Sandbox",
    "version": "1.0",
    "features": ["code_execution", "isolation", "safety"],
    "timestamp": "2025-10-22"
}

# 序列化
json_str = json.dumps(data, indent=2)
print("JSON 数据:")
print(json_str)

# 反序列化
parsed = json.loads(json_str)
print(f"\\n解析后的名称: {parsed['name']}")
print(f"特性数量: {len(parsed['features'])}")
"""
                },
                resource_uri=None,
                prompt_name=None,
                prompt_arguments=None,
                timeout=30,
                metadata={
                    "session_id": session_id1
                }
            ),
        ]
        
        # 顺序执行测试任务
        for i, task in enumerate(tasks):
            logging.info(f"\n{'='*70}")
            logging.info(f"执行任务 {i+1}/{len(tasks)}: {task.request_type.value}")
            if task.tool_name:
                logging.info(f"工具: {task.tool_name}")
            if task.tool_arguments and 'code' in task.tool_arguments:
                code_preview = task.tool_arguments['code'].strip().split('\n')[0]
                if len(code_preview) > 60:
                    code_preview = code_preview[:60] + "..."
                logging.info(f"代码预览: {code_preview}")
            logging.info(f"会话 ID: {task.metadata.get('session_id', 'N/A')[:8]}...")
            logging.info(f"{'='*70}")
            
            self.input_channel.put(task, async_op=True)
            print(f"after put")
            response: MCPResponse = asyncio.run(self.output_channel.get(async_op=True).async_wait())
            print(f"after get")
            
            if response.success:
                logging.info(f"✓ 任务 {i+1} 成功")
                if response.result:
                    if 'output' in response.result:
                        logging.info(f"输出:\n{response.result['output']}")
                    elif 'tools' in response.result:
                        tools = response.result['tools']
                        logging.info(f"可用工具数量: {len(tools)}")
                        for tool in tools:
                            logging.info(f"  - {tool['name']}: {tool.get('description', 'N/A')}")
                    else:
                        logging.info(f"结果: {response.result}")
                if hasattr(response, 'execution_time'):
                    logging.info(f"执行时间: {response.execution_time:.3f}s")
            else:
                logging.error(f"✗ 任务 {i+1} 失败")
                logging.error(f"错误: {getattr(response, 'error_message', 'Unknown error')}")
            
            # 添加短暂延迟以便观察输出
            time.sleep(0.3)
        
        logging.info(f"\n{'='*70}")
        logging.info("所有测试任务完成")
        
        
        logging.info(f"{'='*70}\n")
