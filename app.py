from pathlib import Path

import gradio as gr
import torch
import os
import json

from demo import YMCA

current_dir = os.path.dirname(os.path.abspath(__file__))
with open(Path(current_dir, "source/test_caption.json")) as f: 
    captions = json.load(f)

EXAMPLE_LIST = [
    [str(Path(current_dir, "source/video7061.mp4")), captions["caption_7061"][0]],
    [str(Path(current_dir, "source/video7118.mp4")), captions["caption_7118"][0]]
]

print(captions["caption_7061"][0])

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    servicer = YMCA(device)
    css = servicer.get_css()

    with gr.Blocks(theme='nota-ai/theme', css=css) as demo:
        header = gr.Markdown(Path('docs/header.md').read_text())
        gr.Markdown(Path('docs/description.md').read_text())
        with gr.Row():
            with gr.Column():
                video = gr.Video(label="Upload a video file", height=400)
                text = gr.Textbox(value="", label="Caption of Selected Video")
                inference_button = gr.Button(value="Inference with YMCA", variant="primary")

                with gr.Tab("Example Videos"):
                    examples = gr.Examples(examples=EXAMPLE_LIST, inputs=video, outputs=[video, text])
            with gr.Column():
                result = gr.Label()

demo.launch()