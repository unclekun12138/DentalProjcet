"""
Clinical Dental Image Analysis Agent - LLM Core
Real Function Calling loop: LLM -> tool_call -> execute -> tool_result -> LLM -> final answer.
"""
import json
from llm_client import LLMClient, SYSTEM_PROMPT


class DentalAnalysisAgent:
    """
    Agent that analyzes dental images using LLM + real tool calling.
    Tools are currently mock (return simulated data) but the FC framework is real.
    """

    def __init__(self):
        self.llm = LLMClient()
        self.tools = self._define_tools()
        self.conversation_history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self.uploaded_file = None
        self.last_report = None  # store latest generated report

    # ------------------------------------------------------------------ tools
    def _define_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "segment_pointcloud",
                    "description": "对上传的口扫点云文件进行牙齿分割，返回分割结果（牙齿数量、牙列标签、分割指标等）",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "点云文件路径（.ply / .obj / .pcd）"
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "generate_report",
                    "description": "根据分割分析数据生成结构化的口腔影像分析报告",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "analysis_data": {
                                "type": "string",
                                "description": "分割结果数据（JSON 字符串或文本摘要）"
                            }
                        },
                        "required": ["analysis_data"]
                    }
                }
            }
        ]

    # --------------------------------------------------------------- session
    def set_file(self, file_path: str):
        """Record the uploaded file path so the LLM can reference it."""
        self.uploaded_file = file_path
        self.conversation_history.append({
            "role": "system",
            "content": f"用户已上传文件: {file_path}。用户可能要求分析此文件。"
        })

    def reset(self):
        """Reset conversation history."""
        self.conversation_history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self.uploaded_file = None
        self.last_report = None

    # ------------------------------------------------------------ main loop
    def chat(self, user_message: str) -> str:
        """
        Multi-turn chat with real function calling.
        Returns the final assistant text reply.
        Side effect: self.last_report may be set if generate_report was called.
        """
        # If user didn't specify a file but we have one uploaded, inject it
        if self.uploaded_file and "file_path" not in user_message:
            user_message = f"{user_message}\n（已上传文件路径: {self.uploaded_file}）"

        self.conversation_history.append({"role": "user", "content": user_message})

        max_turns = 6
        for turn in range(max_turns):
            resp = self.llm.chat(self.conversation_history, tools=self.tools)

            if resp.get("tool_calls"):
                # Append assistant message that contains tool_calls
                self.conversation_history.append(resp)

                for tc in resp["tool_calls"]:
                    fname = tc["function"]["name"]
                    try:
                        fargs = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        fargs = {}

                    print(f"[Agent] >>> Tool call: {fname}({fargs})")
                    result = self._execute_tool(fname, fargs)
                    result_str = json.dumps(result, ensure_ascii=False)
                    print(f"[Agent] <<< Tool result: {result_str[:200]}")

                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result_str,
                    })
            else:
                # No more tool calls -> final answer
                self.conversation_history.append(
                    {"role": "assistant", "content": resp["content"]}
                )
                return resp["content"]

        return "已达到最大工具调用轮次，请简化您的问题后重试。"

    # ------------------------------------------------------------- execution
    def _execute_tool(self, name: str, args: dict) -> dict:
        if name == "segment_pointcloud":
            return self._segment_pointcloud(args.get("file_path", ""))
        elif name == "generate_report":
            return self._generate_report(args.get("analysis_data", ""))
        return {"status": "error", "message": f"未知工具: {name}"}

    # --------------------------------------------------------------- mocks
    def _segment_pointcloud(self, file_path: str) -> dict:
        """Mock point cloud segmentation (framework real, data simulated)."""
        return {
            "status": "success",
            "tool": "segment_pointcloud",
            "file_path": file_path,
            "data_type": "intraoral_scan_pointcloud",
            "teeth_detected": 28,
            "gum_region_detected": True,
            "segmentation_miou": 0.91,
            "boundary_iou": 0.87,
            "tooth_labels_fdi": [
                "11","12","13","14","15","16","17","18",
                "21","22","23","24","25","26","27","28",
                "31","32","33","34","35","36","37","38",
                "41","42","43","44","45","46","47","48"
            ],
            "notes": "Mock 分割结果，接入真实模型后将输出逐点标签。"
        }

    def _generate_report(self, analysis_data: str) -> dict:
        """Mock report generation (framework real, content simulated)."""
        report_md = f"""## 口腔影像智能分析报告

### 分析数据摘要
{analysis_data[:300] if analysis_data else '（无详细数据）'}

### 发现
- 牙齿数量：约 28 颗恒牙
- 牙列完整度：良好
- 牙龈区域：已识别
- 咬合关系：待临床确认

### 分割质量指标
- mIoU: 0.91（mock）
- Boundary IoU: 0.87（mock）

### 临床建议
- 建议结合口内检查进一步评估
- 若有种植/正畸需求，建议完善 CBCT 检查

---
*本报告由 AI 自动生成，仅供参考，不能替代专业医生诊断。*
"""
        self.last_report = report_md
        return {
            "status": "success",
            "tool": "generate_report",
            "report_markdown": report_md,
        }


# --------------------------------------------------------------------- CLI test
if __name__ == "__main__":
    agent = DentalAnalysisAgent()

    print("=" * 50)
    print("Test 1: plain chat (no tool call)")
    print("=" * 50)
    r1 = agent.chat("你好，请简单介绍一下你的功能")
    print("Reply:", r1)

    print("\n" + "=" * 50)
    print("Test 2: file analysis (should trigger tool calls)")
    print("=" * 50)
    agent.set_file("/data/test_scan.ply")
    r2 = agent.chat("请帮我分析这个口扫点云文件")
    print("Reply:", r2)
    print("\nLast report:", agent.last_report[:200] if agent.last_report else "None")
