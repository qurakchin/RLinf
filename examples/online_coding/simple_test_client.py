import json
from httpx import Request, Response
import httpx
import asyncio
import time
import uuid
from datetime import datetime

def agenerate(prefix="", suffix=""):
    TARGET_URL = "http://127.0.0.1:8081/v1/completions"

    # 获取原始请求的头部和主体
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer test-token',
    }
    body = {
        "model": "test-model",
        "prompt": f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>",
        "max_tokens": 50,
        "temperature": 0.7,
        "stream": False,
    }
    print(f'zcy_dbg: 1, headers: {json.dumps(headers, ensure_ascii=False, indent=2)}, body: {json.dumps(body, ensure_ascii=False, indent=2)}')

    # 检查是否启用流式输出

    # 非流式响应处理
    with httpx.Client() as client:
        response = client.post(
            TARGET_URL,
            headers=headers,
            json=body,
            timeout=15.0,
        )
        print(f'zcy_dbg: 2, response: {response.json()}')
        return response.json()['choices'][0]['text']

def atrack(prefix="", suffix="", completion="", accepted=True):
    TARGET_URL = "http://127.0.0.1:8082/api/training/submit"

    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer test-token',
    }

    body = {
        "completionId": str(uuid.uuid4()),
        "filepath": "file:///Users/qurakchin/.vscode/extensions/continue.continue-1.2.3-darwin-arm64/continue_tutorial.py",
        "prefix": prefix,
        "suffix": suffix,
        "prompt": f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>",
        "completion": completion,
        "modelProvider": "openai",
        "modelName": "Qwen2.5-Coder-1.5B-Q8_0.gguf",
        "accepted": accepted,
        "timestamp": datetime.now().isoformat(),
        "time": 4294,
        "uniqueId": str(uuid.uuid4()),
        "numLines": 1,
        "cacheHit": False,
    }

    # 非流式响应处理
    with httpx.Client() as client:
        response = client.post(
            TARGET_URL,
            headers=headers,
            json=body,
            timeout=15.0,
        )
        print(f'zcy_dbg: 2, response: {response.json()}')

def loop():
    prefix = "if x[j] > x[j + 1]:\n                x[j], x[j + 1] = x[j + 1], x[j]\n    return x\n\ndef han"
    suffix = "\n# —————————————————————————————————————————————————     Agent      ————————————————————————————————————————————————— #\n#           Agent equips the Chat model with the tools needed to handle a wide range of coding tasks, allowing\n#           the model to make decisions and save you the work of manually finding context and performing actions.\n\n# 1. Switch from \"Chat\" to \"Agent\" mode using the dropdown in the bottom left of the input box"
    for i in range(10):
        print(f'zcy_dbg: loop: i={i}')
        time.sleep(0.001)
        completion = agenerate(prefix="prefix", suffix="suffix")
        time.sleep(0.001)
        atrack(prefix="prefix", suffix="suffix", completion=completion, accepted=True)

if __name__ == "__main__":
    loop()
