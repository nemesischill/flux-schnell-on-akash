import gradio as gr
import numpy as np
import random
import torch
from diffusers import DiffusionPipeline
from dynaconf import Dynaconf
from loguru import logger
import os

settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['settings.yaml'],
    environments=True,
    load_dotenv=True,
)

device = torch.device(settings.model.device if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16

if not os.path.exists(settings.model.cache_dir):
    logger.info(f"Downloading model weights (black-forest-labs/FLUX.1-schnell)")

pipe = DiffusionPipeline.from_pretrained(
    settings.model.id,
    torch_dtype=dtype,
    use_safetensors=True,
    add_watermarker=False,
    cache_dir=settings.model.cache_dir,
    low_cpu_mem_usage=True
).to(device)

logger.info(f"Weights downloaded (black-forest-labs/FLUX.1-schnell)")

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048


def infer(prompt, seed=42, randomize_seed=False, width=1024, height=1024, num_inference_steps=4,
          progress=gr.Progress(track_tqdm=True)):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator().manual_seed(seed)
    image = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        generator=generator,
        guidance_scale=0.0
    ).images[0]
    return image, seed


examples = [
    "a tiny astronaut hatching from an egg on the moon",
    "a cat holding a sign that says hello world",
    "an anime illustration of a wiener schnitzel",
]

css = """
#col-container {
    margin: 0 auto;
    max-width: 520px;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(f"""# FLUX.1 [schnell]
12B param rectified flow transformer distilled from [FLUX.1 [pro]](https://blackforestlabs.ai/) for 4 step generation
[[blog](https://blackforestlabs.ai/announcing-black-forest-labs/)] [[model](https://huggingface.co/black-forest-labs/FLUX.1-schnell)]
        """)

        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )

            run_button = gr.Button("Run", scale=0)

        result = gr.Image(label="Result", show_label=False)

        with gr.Accordion("Advanced Settings", open=False):
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )

            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

            with gr.Row():
                width = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024,
                )

                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024,
                )

            with gr.Row():
                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=4,
                )

        gr.Examples(
            examples=examples,
            fn=infer,
            inputs=[prompt],
            outputs=[result, seed],
            cache_examples="lazy"
        )

    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=infer,
        inputs=[prompt, seed, randomize_seed, width, height, num_inference_steps],
        outputs=[result, seed]
    )

if __name__ == "__main__":
    demo.queue(max_size=20).launch(server_name="0.0.0.0", server_port=7860)
