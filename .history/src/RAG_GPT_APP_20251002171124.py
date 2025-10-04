import gradio as gr
from utils.upload_file import UploadFile
from utils.chatbot import ChatBot
from utils.ui_settings import UISettings


with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("RAG-GPT"):
            
            # First ROW:
            with gr.Row() as row_one:
                with gr.Column(visible=False) as reference_bar:
                    ref_output = gr.Markdown()

                with gr.Column() as chatbot_output:
                    chatbot = gr.Chatbot(
                        [],
                        type="messages",
                        elem_id="chatbot",
                        height=400,
                        avatar_images=(
                            ("images/user.png"), "images/ai_agent.png"),
                    )
                    
            # SECOND ROW:
            with gr.Row():
                input_txt = gr.Textbox(
                    lines=2,
                    scale=8,
                    placeholder="Enter text and press enter, or upload PDF files",
                    container=False,
                )

            # Third ROW:
            with gr.Row() as row_two:
                text_submit_btn = gr.Button(value="Submit text")
                sidebar_state = gr.State(False)
                btn_toggle_sidebar = gr.Button(
                    value="References")
                btn_toggle_sidebar.click(UISettings.toggle_sidebar, [sidebar_state], [
                    reference_bar, sidebar_state])
                
                upload_btn = gr.UploadButton(
                    "📁 Upload PDF or doc files", file_types=[
                        '.pdf',
                        '.doc'
                    ],
                    file_count="multiple")
                rag_with_dropdown = gr.Dropdown(
                    label="RAG with", choices=["Preprocessed doc", "Upload doc: Process for RAG"], value="Preprocessed doc")
                clear_button = gr.ClearButton([input_txt, chatbot])

            # Process:
            file_msg = upload_btn.upload(fn=UploadFile.uploads_files, inputs=[
                    upload_btn, chatbot, rag_with_dropdown], outputs=[input_txt, chatbot], queue=False)
            
            txt_msg = input_txt.submit(fn=ChatBot.response,
                                    inputs=[chatbot, input_txt,
                                            rag_with_dropdown],
                                    outputs=[input_txt,
                                                chatbot, ref_output],
                                    queue=False).then(lambda: gr.Textbox(interactive=True),
                                                        None, [input_txt], queue=False)

            txt_msg = text_submit_btn.click(fn=ChatBot.response,
                                            inputs=[chatbot, input_txt,
                                                    rag_with_dropdown],
                                            outputs=[input_txt,
                                                    chatbot, ref_output],
                                            queue=False).then(lambda: gr.Textbox(interactive=True),
                                                            None, [input_txt], queue=False)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)