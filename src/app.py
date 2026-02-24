import gradio as gr
import torch
from PIL import Image
from src.config import Config
from src.model import get_model
import numpy as np
import src.vars as var


def get_model_and_transform():
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps:0"
    else:
        device = "cpu"
    model_config = Config.from_json(var.MODEL_CONFIG_PATH)
    model, image_transforms = get_model(model_config, device=device, test_mode=True)
    return model, image_transforms, device


model, image_transform, device = get_model_and_transform()


# Functions
@torch.no_grad()
def classify_image(image_fp: str):
    image = Image.open(image_fp)
    image = image.convert("RGB")
    image = np.array(image)
    image = image_transform(image)
    image = image.to(device)

    logits = model(image.unsqueeze(0))
    label = torch.argmax(logits, dim=-1).cpu().item()

    label = str(int(label) + 1)
    return label


with gr.Blocks() as demo:
    with gr.Tab("分類"):
        with gr.Row():
            with gr.Column():
                classify_btn = gr.Button("分類する")
                classification_input = gr.Image(type="filepath")
            with gr.Column():
                classification_output = gr.Label()

    classify_btn.click(
        classify_image, inputs=[classification_input], outputs=[classification_output]
    )

demo.launch()
