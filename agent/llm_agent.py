
"""
Clinical Dental Image Analysis Agent - LLM Core
Uses LLM Function Calling to orchestrate segmentation and analysis
"""
import json
import re


class DentalAnalysisAgent:
    """
    Agent that analyzes dental images using LLM + tool calling.
    Currently supports mock tools; will connect to real segmentation models.
    """
    
    def __init__(self):
        self.tools = self._define_tools()
        self.conversation_history = []
    
    def _define_tools(self):
        """Define available tools the LLM can call."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "segment_cbct",
                    "description": "Run tooth segmentation on CBCT volume data",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "description": "Path to CBCT NIfTI file"}
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "segment_pointcloud",
                    "description": "Run tooth segmentation on intraoral scan point cloud",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "description": "Path to point cloud PLY/OBJ file"}
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "generate_report",
                    "description": "Generate structured dental analysis report",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "segmentation_result": {"type": "string", "description": "Segmentation results summary"},
                            "patient_info": {"type": "string", "description": "Patient info if available"}
                        },
                        "required": ["segmentation_result"]
                    }
                }
            }
        ]
    
    def analyze(self, user_message, file_path=None):
        """
        Main analysis entry point.
        1. Determine data type
        2. Call appropriate segmentation tool
        3. Generate report
        """
        # Mock implementation - will connect to real LLM API
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # Step 1: Determine data type
        data_type = self._detect_data_type(file_path)
        
        # Step 2: Run segmentation (mock)
        if data_type == "cbct":
            result = self._mock_segment_cbct(file_path)
        elif data_type == "pointcloud":
            result = self._mock_segment_pointcloud(file_path)
        else:
            result = {"error": "Unknown data type"}
        
        # Step 3: Generate report
        report = self._generate_report(result, data_type)
        
        self.conversation_history.append({"role": "assistant", "content": report})
        return report
    
    def _detect_data_type(self, file_path):
        """Detect whether input is CBCT or point cloud based on file extension."""
        if file_path is None:
            return "unknown"
        ext = file_path.lower().split('.')[-1]
        if ext in ('nii', 'nii.gz', 'nrrd', 'mha'):
            return "cbct"
        elif ext in ('ply', 'obj', 'pcd', 'stl'):
            return "pointcloud"
        return "unknown"
    
    def _mock_segment_cbct(self, file_path):
        """Mock CBCT segmentation - will be replaced with real model."""
        return {
            "data_type": "CBCT",
            "teeth_detected": 28,
            "missing_teeth": [],
            "segmentation_dice": 0.96,  # mock
            "structures": ["teeth", "mandible", "maxilla", "root_canals"]
        }
    
    def _mock_segment_pointcloud(self, file_path):
        """Mock point cloud segmentation - will be replaced with real model."""
        return {
            "data_type": "intraoral_scan",
            "teeth_detected": 28,
            "gum_detected": True,
            "segmentation_miou": 0.91,  # mock
            "boundary_accuracy": 0.85
        }
    
    def _generate_report(self, result, data_type):
        """Generate structured analysis report."""
        if "error" in result:
            return f"分析失败：{result['error']}"
        
        report = f"""## 口腔影像分析报告

### 数据类型
{result.get('data_type', 'Unknown')}

### 分割结果
- 检测牙齿数量：{result.get('teeth_detected', 'N/A')} 颗
- 分割指标：Dice={result.get('segmentation_dice', result.get('segmentation_miou', 'N/A'))}

### 结构识别
{chr(10).join(f'- {s}' for s in result.get('structures', ['牙齿', '牙龈']))}

### 临床建议
- 建议结合临床检查进一步评估
- 本报告由 AI 自动生成，仅供参考
"""
        return report
    
    def chat(self, user_message):
        """Multi-turn chat interface."""
        self.conversation_history.append({"role": "user", "content": user_message})
        # Mock response - will connect to real LLM
        response = f"收到您的问题：{user_message}\n当前为 MVP 版本，接入真实模型后将提供专业解答。"
        self.conversation_history.append({"role": "assistant", "content": response})
        return response

