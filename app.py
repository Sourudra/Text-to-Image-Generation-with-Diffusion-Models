import gradio as gr
import numpy as np
import random
import torch
from diffusers import DiffusionPipeline

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load your DiffusionPipeline model
model_repo_id = "stabilityai/sdxl-turbo"
pipe = DiffusionPipeline.from_pretrained(model_repo_id, torch_dtype=torch_dtype)
pipe = pipe.to(device)

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024

# Define the custom model inference function
def custom_infer(
    prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps
):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator().manual_seed(seed)

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        generator=generator,
    ).images[0]

    return image, seed


# Gradio interface for custom model
def custom_model_ui():
    with gr.Blocks() as custom_demo:
        gr.Markdown("## Needs a GPU for best performance and it is highly customizable.\n ## stabilityai/sdxl-turbo")
        #gr.Markdown('<p style="font-size: 30px;">Needs a GPU for best performance and it is highly customizable.</p>\n<p style="font-size: 50px; font-weight: bold;">stabilityai/sdxl-turbo</p>', unsafe_allow_html=True)



        with gr.Row():
            prompt = gr.Text(label="Prompt")
            run_button = gr.Button("Generate")

        result = gr.Image(label="Generated Image")
        negative_prompt = gr.Text(label="Negative Prompt", placeholder="Optional")
        seed = gr.Slider(0, MAX_SEED, label="Seed", step=1, value=0)
        randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
        width = gr.Slider(256, MAX_IMAGE_SIZE, step=32, value=1024, label="Width")
        height = gr.Slider(256, MAX_IMAGE_SIZE, step=32, value=1024, label="Height")
        guidance_scale = gr.Slider(0, 10, step=0.1, value=7.5, label="Guidance Scale")
        num_inference_steps = gr.Slider(1, 50, step=1, value=30, label="Inference Steps")

        run_button.click(
            custom_infer,
            inputs=[prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps],
            outputs=[result, seed],
        )

    return custom_demo


# Preloaded Gradio model
def preloaded_model_ui():
    with gr.Blocks() as preloaded_demo:
        gr.Markdown("## Works well on CPU and it is faster.")
        preloaded_demo = gr.load("models/ZB-Tech/Text-to-Image")

    return preloaded_demo


# Combine both interfaces in tabs
with gr.Blocks() as demo:
    with gr.Tab("Quick Image Generation"):
        preloaded_ui = preloaded_model_ui()

    with gr.Tab("Advanced Image Generation"):
        custom_ui = custom_model_ui()

if __name__ == "__main__":
    demo.launch()