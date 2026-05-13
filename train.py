import argparse
import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from datetime import datetime, timedelta
import importlib


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def denorm(x):
    return (x + 1.0) / 2.0

def format_time(seconds):
    delta = timedelta(seconds=seconds)
    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{days}d {hours}h {minutes}m {seconds}s'

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dir", type=str, default="CoCo/coco_train/train_2017")
    parser.add_argument("--batch_size", type=int, default=4)

    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=1000000)
    parser.add_argument("--encoder_module", type=str, default="encoder2")
    parser.add_argument("--decoder_module", type=str, default="decoder24_with_graph")


    parser.add_argument("--img_loss", type=str, default='both', choices=['lpips', 'mse', 'both'])
    parser.add_argument("--lambda_mse", type=float, default=1.0)
    parser.add_argument("--lambda_lpips", type=float, default=0.25)

    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--save_model_freq", type=int, default=100000)
    parser.add_argument("--save_image_freq", type=int, default=1000)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_accum_steps", type=int, default=32)
    parser.add_argument("--resume", type=str, default=None)


    return parser


def main(params):
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    random.seed(params.seed)

    now = datetime.now()
    dt_string = now.strftime("%y%m%d%H%M%S")
    decoder_name = params.decoder_module.split('.')[-1]
    exp_name = f"nosier_{dt_string}_{decoder_name}_bs{params.batch_size}_ga{params.grad_accum_steps}_{params.img_loss}_{params.lambda_mse}_{params.lambda_lpips}_lr{params.lr}"

    exp_path = os.path.join("runs", exp_name)
    model_dir = os.path.join(exp_path, "models")
    img_dir = os.path.join(exp_path, "imgs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    with open(os.path.join(exp_path, 'args.txt'), 'w') as f:
        for arg in vars(params):
            f.write(f"{arg}: {getattr(params, arg)}\n")
        f.write(f"Command: {' '.join(sys.argv)}\n")

    normalize_vqgan = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize (x - 0.5) / 0.5
    transform = transforms.Compose([
        lambda img: transforms.Resize(params.img_size)(img) if min(img.size) < params.img_size else img,
        transforms.RandomCrop(params.img_size),
        transforms.ToTensor(),
        normalize_vqgan
    ])
    dataset = ImageFolder(params.train_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=params.batch_size, shuffle=True, num_workers=8)


    decoder_module = importlib.import_module(params.decoder_module)
    Decoder_class = getattr(decoder_module, "Decoder")

    encoder_module = importlib.import_module(params.encoder_module)
    Encoder_class = getattr(encoder_module, "Encoder")

    encoder = Encoder_class().to(device)
    decoder = Decoder_class().to(device)

    import lpips
    img_lpips = lpips.LPIPS(net='vgg').to(device)
    img_mse = nn.MSELoss()


    model_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(model_params, lr=params.lr)


    if params.resume is not None and os.path.isfile(params.resume):
        checkpoint = torch.load(params.resume, map_location=device)
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


    start_time = time.time()
    data_iter = iter(loader)

    for step in range(1, params.steps + 1):
        try:
            imgs, _ = next(data_iter)
        except:
            data_iter = iter(loader)
            imgs, _ = next(data_iter)
        imgs = imgs.to(device)

        latents = encoder(imgs)
        recon_imgs, _ = decoder(latents)

        loss_lpips_val = None
        loss_mse_val = None

        if params.img_loss == 'lpips':
            loss_lpips_val = img_lpips(imgs, recon_imgs).mean()
            loss = loss_lpips_val

        elif params.img_loss == 'mse':
            loss_mse_val = img_mse(imgs, recon_imgs)
            loss = loss_mse_val

        elif params.img_loss == 'both':
            loss_lpips_val = img_lpips(imgs, recon_imgs).mean()
            loss_mse_val = img_mse(imgs, recon_imgs)
            loss = params.lambda_mse * loss_mse_val + params.lambda_lpips * loss_lpips_val
        else:
            raise ValueError(f"Invalid loss type: {params.img_loss}")

        loss = loss / params.grad_accum_steps  # Normalize loss
        loss.backward()

        if step % params.grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % params.log_freq == 0:
            duration = time.time() - start_time
            log_msg = f"{dt_string} [{step:07d}] "

            if params.img_loss == 'lpips':
                log_msg += f"LPIPS Loss: {loss_lpips_val.item():.5f} "
            elif params.img_loss == 'mse':
                log_msg += f"MSE Loss: {loss_mse_val.item():.5f} "
            elif params.img_loss == 'both':
                log_msg += (
                    f"LPIPS: {loss_lpips_val.item():.5f} "
                    f"MSE: {loss_mse_val.item():.5f} "
                    f"Total Loss: {loss.item():.5f} "
                )

            log_msg += f"Time: {format_time(duration)}"
            print(log_msg)

            with open(os.path.join(exp_path, 'logs.txt'), 'a') as f:
                f.write(log_msg + "\n")

        if step % params.save_image_freq == 0:
            with torch.no_grad():
                imgs_vis = imgs[:4]  
                recons_vis = recon_imgs[:4]

                grid = torch.cat([denorm(imgs_vis), denorm(recons_vis)], dim=0)  
                save_path = os.path.join(img_dir, f"recon_step{step:07d}.jpg")
                save_image(grid, save_path, nrow=4)

        if step % params.save_model_freq == 0:
            torch.save({
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step
            }, os.path.join(model_dir, f"checkpoint-{step:06d}.pth"))

if __name__ == "__main__":
    parser = get_parser()
    params = parser.parse_args()
    main(params)