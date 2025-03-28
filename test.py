import torch
from utils import DTYPE
from diffusers import StableDiffusionPipeline, LCMScheduler
import argparse


def test(args):
    scheduler = LCMScheduler.from_pretrained(
        args.model_name,
        subfolder= "scheduler"
    )
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.model_name,
        scheduler=scheduler,
        torch_dtype=DTYPE,
        safety_checker=None
    ).to(args.device)

    pipeline.unet.load_state_dict(torch.load(args.checkpoint_path, weights_only=True))

    generator = torch.Generator(device=args.device).manual_seed(42)
    image = pipeline(
        prompt = [args.prompt],
        num_inference_steps = 4,
        num_images_per_prompt = 1,
        generator=generator,
        guidance_scale = 8.0,
    ).images[0]
    image.save("./result.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=False, default="sd-legacy/stable-diffusion-v1-5")
    parser.add_argument("--checkpoint_path", required=False, default="./checkpoint/2_model.pth")
    parser.add_argument("--device", required=False, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prompt", required=False, default="Cat")
    arguments = parser.parse_args()
    test(arguments)