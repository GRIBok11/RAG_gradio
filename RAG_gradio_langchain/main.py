import gradio as gr
from chatbot import add_message, bot, clear_history

request_count = 0
max_requests = 20
clear_history()

def get_request_count():
    global request_count, max_requests
    return update_request_count(request_count, max_requests)

with gr.Blocks(fill_height=True) as demo:
    chatbot = gr.Chatbot(
        elem_id="chatbot",
        bubble_full_width=False,
        scale=1,
    )

    chat_input = gr.MultimodalTextbox(interactive=True,
                                      file_count="multiple",
                                      placeholder="Enter message or upload file...", show_label=False)
    clear_button = gr.Button("🗑", size="sm")


    chat_msg = chat_input.submit(add_message, [chatbot, chat_input], [chatbot, chat_input])
    bot_msg = chat_msg.then(bot, chatbot, chatbot, api_name="bot_response")
    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

    clear_button.click(clear_history, [], [chatbot, chat_input])




demo.launch(share=True)