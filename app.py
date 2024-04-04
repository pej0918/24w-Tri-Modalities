from pathlib import Path

import gradio as gr
import torch

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with gr.Blocks(theme='nota-ai/theme') as demo:
        gr.Markdown(Path('docs/header.md').read_text())
        gr.Markdown(Path('docs/description.md').read_text())
        with gr.Row():
            with gr.Column():
                # Get the video files from the user
                video = gr.Video(label="Upload a video file")

demo.launch()