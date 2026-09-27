
"""
LLM Client - OpenAI-compatible API (DeepSeek / Qwen / etc.)
Usage:
    from llm_client import LLMClient
    llm = LLMClient(api_key="your-key", base_url="https://api.deepseek.com/v1")
    response = llm.chat("你好")
"""
import os
import json
from typing import List, Dict, Optional


class LLMClient:
    """OpenAI-compatible LLM client."""
    
    def __init__(
        self,
        api_key: str = None,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
    ):
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.base_url = base_url
        self.model = model
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize OpenAI client if api_key is available."""
        if not self.api_key:
            print("[LLM] No API key set, running in mock mode")
            return
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            print(f"[LLM] Connected to {self.base_url}, model={self.model}")
        except ImportError:
            print("[LLM] openai package not installed, running in mock mode")
    
    def chat(self, messages: List[Dict], tools: List[Dict] = None, temperature: float = 0.7) -> str:
        """
        Chat with LLM.
        Args:
            messages: [{"role": "user/assistant/system", "content": "..."}]
            tools: optional function definitions for tool calling
            temperature: creativity
        Returns:
            assistant response text
        """
        if self.client is None:
            return self._mock_response(messages)
        
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            }
            if tools:
                kwargs["tools"] = tools
            
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"[LLM Error] {str(e)}"
    
    def _mock_response(self, messages: List[Dict]) -> str:
        """Mock response when no API key."""
        last_msg = messages[-1]["content"] if messages else ""
        return f"[Mock LLM] 收到您的问题：{last_msg[:100]}\n配置 API key 后即可获得真实回答。"


class DentalRAG:
    """Simple RAG for dental medical knowledge."""
    
    def __init__(self):
        # Basic dental knowledge base
        self.knowledge = {
            "tooth_numbering": "FDI编号系统：右上18-11，左上21-28，右下48-41，左下31-38",
            "cbct_indications": "CBCT用于：种植牙术前评估、阻生牙定位、根管治疗、颌骨病变诊断",
            "ios_indications": "口扫用于：正畸、修复、种植、咬合分析、隐形牙套设计",
            "common_issues": "常见口腔问题：龋齿、牙周炎、智齿阻生、牙列不齐、颞下颌关节紊乱",
        }
    
    def retrieve(self, query: str) -> str:
        """Retrieve relevant knowledge."""
        results = []
        for key, value in self.knowledge.items():
            if any(word in query for word in key.split("_")):
                results.append(value)
        if not results:
            return "未找到相关知识，请咨询专业医生。"
        return "\n".join(results)


# System prompt for dental analysis agent
SYSTEM_PROMPT = """你是一个专业的口腔影像分析助手。你的职责是：
1. 分析用户上传的口腔影像（CBCT或口扫点云）
2. 识别牙齿数量、位置和状态
3. 生成结构化的分析报告
4. 回答用户关于口腔健康的问题

注意：
- 你的分析仅供参考，不能替代医生诊断
- 如果不确定，建议用户咨询专业牙医
- 使用专业但易懂的语言
"""


if __name__ == "__main__":
    # Test
    llm = LLMClient()
    print(llm.chat([{"role": "user", "content": "什么是FDI编号系统？"}]))
    
    rag = DentalRAG()
    print("\nRAG test:", rag.retrieve("tooth numbering"))

