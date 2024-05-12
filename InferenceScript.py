from diffusers import DDIMPipeline
import torch
from diffusers.models import AutoencoderKL
from torchvision import transforms as tfms
import numpy as np
from PIL import Image
import argparse
import inspect



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    parser.add_argument(
        "--out_name",
        type=str,
        default="Image.png",
        help="The config of the Dataset, leave as None if there's only one config.",
    )

    parser.add_argument(
        "--pipeline_dir",
        type=str,
        default= None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )

    args = parser.parse_args()

    if args.pipeline_dir is None and args.out_name is None:
        raise ValueError("You must specify path of diffusion pipeline.")



    return args


vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae",torch_dtype=torch.float16).to("cuda")

def latent2img(latents):
  latent_image = (1 / 0.18215) * latents
  with torch.no_grad():
    image = vae.decode(latent_image).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
  return images

def Inference_Script(scheduler,unet):
  generator = None
  latents = torch.randn(1,4,32,32)
  latents = latents.to("cuda",torch.float16)
  scheduler.set_timesteps(1000)
  eta: float = 0.0
          # scale the initial noise by the standard deviation required by the scheduler
  #scheduler.set_timesteps(10)
  latents = latents * scheduler.init_noise_sigma
          # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
  accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())

  extra_kwargs = {}
  if accepts_eta:
    extra_kwargs["eta"] = eta

  with torch.no_grad():
    for t in (scheduler.timesteps):
      image = []
      latent_model_input = scheduler.scale_model_input(latents, t)
                # predict the noise residual
      noise_prediction = unet(latent_model_input, t).sample
                # compute the previous noisy sample x_t -> x_t-1
      latents = scheduler.step(noise_prediction, t, latents, **extra_kwargs).prev_sample

            # decode the image latents with the VAE
      image = latent2img(latents.to(torch.float16))

    return image#ImagePipelineOutput(images=image)
  

def main(args):
    pipeline = DDIMPipeline.from_pretrained(args.pipeline_dir)
    unet = pipeline.unet.to("cuda").to(torch.float16)
    scheduler = pipeline.scheduler
    img = Inference_Script(scheduler,unet)
    img = Image.fromarray(img[0], "RGB")
    img.save(args.out_name)


if __name__ == "__main__":
    args = parse_args()
    main(args)