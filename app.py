import gradio as gr
import asyncio
from trial_with_memory import chat

# Helper to run async LangGraph chat() inside Gradio
def run_async(coro):
    return asyncio.run(coro)

def chatbot_interface(user_message, chat_history):
    resp = run_async(chat(user_message, thread_id="gradio-thread-1"))

    sql = resp.get("sql")
    result = resp.get("result")
    error = resp.get("error")

    # Build reply
    if error:
        bot_reply = f"❌ **Error:** {error}"
    else:
        reply = f"### ✅ SQL Executed\n```\n{sql}\n```"

        # Format result table
        if result and result.get("rows"):
            cols = result["columns"]
            rows = result["rows"]

            table_txt = "### 📊 Result\n"
            table_txt += "| " + " | ".join(cols) + " |\n"
            table_txt += "| " + " | ".join(['---'] * len(cols)) + " |\n"
            for row in rows[:50]:
                table_txt += "| " + " | ".join([str(x) for x in row]) + " |\n"

            reply += "\n" + table_txt
        else:
            reply += "\nNo rows returned."

        bot_reply = reply

    # IMPORTANT: Convert to message dicts (Gradio messages mode)
    if chat_history is None:
        chat_history = []

    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": bot_reply})

    return "", chat_history


# GRADIO UI
with gr.Blocks(title="Blinkit Text-to-SQL Chatbot") as demo:
    gr.Markdown("""
# 🟢 Blinkit Text-to-SQL Chatbot  
Ask any analytical question about your Blinkit dataset.
""")

    chatbot = gr.Chatbot(height=550)   # default = messages mode
    user_input = gr.Textbox(show_label=False, placeholder="Ask your question...")
    send_btn = gr.Button("Send", variant="primary")

    send_btn.click(
        chatbot_interface,
        inputs=[user_input, chatbot],
        outputs=[user_input, chatbot]
    )

    user_input.submit(
        chatbot_interface,
        inputs=[user_input, chatbot],
        outputs=[user_input, chatbot]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
