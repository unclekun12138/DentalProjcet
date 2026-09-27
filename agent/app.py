"""
Clinical Dental Image Analysis Agent - Gradio UI
File upload + Chatbot (real LLM + Function Calling) + Report display.
"""
import os
import gradio as gr
from llm_agent import DentalAnalysisAgent

# Singleton agent
_agent = None


def get_agent() -> DentalAnalysisAgent:
    global _agent
    if _agent is None:
        _agent = DentalAnalysisAgent()
    return _agent


# ------------------------------------------------------------------ handlers
def on_file_upload(file):
    """When user uploads a file, register it with the agent."""
    if file is None:
        return "未选择文件。"
    agent = get_agent()
    agent.set_file(file.name)
    return f"已上传: {os.path.basename(file.name)}\n路径: {file.name}"


def respond(message, chat_history):
    """Send user message to agent, stream back the reply."""
    if not message.strip():
        return "", chat_history

    agent = get_agent()
    reply = agent.chat(message)

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": reply})

    # Update report area if a report was generated
    report_md = agent.last_report or "（暂无报告。请先上传文件并要求分析。）"
    return "", chat_history, report_md


def new_session():
    """Reset conversation."""
    agent = get_agent()
    agent.reset()
    return [], "（已开启新会话）"


# --------------------------------------------------------------------- UI
with gr.Blocks(title="口腔影像智能分析 Agent") as demo:
    gr.Markdown("# 口腔影像智能分析 Agent")
    gr.Markdown("上传口扫点云文件 → 与 AI 对话分析 → 自动生成报告")

    with gr.Row():
        # ---- left: file upload ----
        with gr.Column(scale=1):
            gr.Markdown("### 📁 文件上传")
            file_input = gr.File(
                label="点云 / 影像文件",
                file_types=[".ply", ".obj", ".pcd", ".stl", ".nii", ".dcm"],
            )
            upload_status = gr.Textbox(label="上传状态", interactive=False, lines=2)
            file_input.change(fn=on_file_upload, inputs=[file_input],
                              outputs=[upload_status])

            gr.Markdown("---")
            reset_btn = gr.Button("🔄 开启新会话", variant="secondary")

        # ---- right: chat + report ----
        with gr.Column(scale=2):
            gr.Markdown("### 💬 对话分析")
            chatbot = gr.Chatbot(
                height=380, type="md",
                label="AI 助手",
            )
            msg_input = gr.Textbox(
                label="输入问题",
                placeholder="例如：帮我分析这个口扫文件 / 这是什么牙齿问题？",
            )

            gr.Markdown("### 📊 分析报告")
            report_output = gr.Markdown("（暂无报告。请先上传文件并要求分析。）")

    # wire up
    msg_input.submit(
        fn=respond,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot, report_output],
    )
    reset_btn.click(fn=new_session, outputs=[chatbot, report_output])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
