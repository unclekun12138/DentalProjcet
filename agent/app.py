
"""
Clinical Dental Image Analysis Agent - MVP
Gradio interface with mock segmentation pipeline
"""
import os
import numpy as np
import gradio as gr
import open3d as o3d

# Mock segmentation - will be replaced with real model
def segment_point_cloud(file_path):
    """Mock segmentation: returns random labels for now."""
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    n = len(points)
    
    # Mock: assign random labels (0=gum, 1-8=teeth)
    labels = np.random.randint(0, 9, size=n)
    
    return points, labels


def analyze_dental_image(file):
    """Main agent function: upload -> segment -> generate report."""
    if file is None:
        return "请上传口扫点云文件（.ply/.obj）", None
    
    try:
        # Step 1: Load and segment
        points, labels = segment_point_cloud(file.name)
        
        # Step 2: Generate analysis report
        n_teeth = len(np.unique(labels)) - 1  # exclude gum
        n_points = len(points)
        
        report = f"""## 口腔影像分析报告

### 基本信息
- 点云点数：{n_points:,}
- 检测到牙齿数量：约 {n_teeth} 颗

### 分析结果
- 牙龈区域：已识别
- 牙齿分割：完成（mock 数据，待接入真实模型）
- 牙列完整性：待评估

### 注意
当前为 MVP 演示版本，分割结果为模拟数据。
接入真实模型后将显示准确的牙齿分割与编号。
"""
        return report, points.tolist()
    
    except Exception as e:
        return f"处理出错：{str(e)}", None


# Build Gradio interface
with gr.Blocks(title="口腔影像智能分析 Agent") as demo:
    gr.Markdown("# 口腔影像智能分析 Agent")
    gr.Markdown("上传口扫点云文件，自动进行牙齿分割与分析报告生成")
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(
                label="上传口扫文件（.ply / .obj）",
                file_types=[".ply", ".obj", ".pcd"]
            )
            analyze_btn = gr.Button("开始分析", variant="primary")
        
        with gr.Column():
            report_output = gr.Markdown(label="分析报告")
    
    analyze_btn.fn = analyze_dental_image
    analyze_btn.inputs = [file_input]
    analyze_btn.outputs = [report_output, gr.State()]


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

