"""
LLM Client - OpenAI-compatible API (DeepSeek / Qwen / etc.)
Supports Function Calling: returns full message with tool_calls.
"""
import os
import json
from typing import List, Dict, Optional


class LLMClient:
    """OpenAI-compatible LLM client with function calling support."""

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
    ):
        self._load_env()
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.base_url = base_url or os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1")
        self.model = model or os.environ.get("LLM_MODEL", "deepseek-chat")
        self.client = None
        self._init_client()

    def _load_env(self):
        """Load .env from the same directory as this file."""
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        if os.path.exists(env_path):
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, val = line.split("=", 1)
                        os.environ.setdefault(key.strip(), val.strip())

    def _init_client(self):
        if not self.api_key:
            print("[LLM] No API key set, running in mock mode")
            return
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            print(f"[LLM] Connected to {self.base_url}, model={self.model}")
        except ImportError:
            print("[LLM] openai package not installed, running in mock mode")

    def chat(self, messages: List[Dict], tools: List[Dict] = None,
             temperature: float = 0.7) -> Dict:
        """
        Chat with LLM. Returns a message dict:
            {
                "role": "assistant",
                "content": str,
                "tool_calls": [ {id, type, function: {name, arguments}} ] or None
            }
        """
        if self.client is None:
            return {
                "role": "assistant",
                "content": self._mock_response(messages),
                "tool_calls": None,
            }

        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            }
            if tools:
                kwargs["tools"] = tools

            response = self.client.chat.completions.create(**kwargs)
            msg = response.choices[0].message

            result = {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": None,
            }

            if getattr(msg, "tool_calls", None):
                result["tool_calls"] = []
                for tc in msg.tool_calls:
                    result["tool_calls"].append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments or "{}",
                        },
                    })

            return result
        except Exception as e:
            return {
                "role": "assistant",
                "content": f"[LLM Error] {str(e)}",
                "tool_calls": None,
            }

    def _mock_response(self, messages: List[Dict]) -> str:
        last = messages[-1].get("content", "") if messages else ""
        return f"[Mock LLM] 收到：{last[:100]}"


# System prompt for dental analysis agent
SYSTEM_PROMPT = """你是一个专业的口腔影像分析助手。你的职责是：
1. 分析用户上传的口腔影像（CBCT或口扫点云）
2. 调用工具进行牙齿分割和分析
3. 根据分割结果生成结构化分析报告
4. 回答用户关于口腔健康的问题

工作流程：
- 当用户上传文件并要求分析时，先调用 segment_pointcloud 工具进行分割
- 分割完成后，调用 generate_report 工具生成分析报告
- 向用户解释分析结果

注意：
- 你的分析仅供参考，不能替代医生诊断
- 使用专业但易懂的语言回答用户
"""


if __name__ == "__main__":
    llm = LLMClient()
    resp = llm.chat([{"role": "user", "content": "你好，介绍一下你自己"}])
    print(resp["content"])
