import torch
import torch.nn as nn
from torchvision import transforms, utils
from PIL import Image
import os

from encoder2 import Encoder
from decoder24_with_graph import Decoder

def denorm(x):
    return (x + 1.0) / 2.0

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = "250708204953-4m.pth"
    input_image_path = "test.png"
    output_dir = "test_results"
    num_tests = 5
    img_size = 256

    os.makedirs(output_dir, exist_ok=True)

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)

    if not os.path.exists(checkpoint_path):
        return

    print(f"{checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])

    encoder.eval()
    decoder.eval()

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    if not os.path.exists(input_image_path):
        return

    img_pil = Image.open(input_image_path).convert('RGB')
    img_tensor = transform(img_pil).unsqueeze(0).to(device)  # (1, 3, H, W)


    with torch.no_grad():
        # Encoder
        latents = encoder(img_tensor)

        for i in range(num_tests):
            # Decoder
            recon_img, arch_info = decoder(latents)

            diff = torch.abs(img_tensor - recon_img)

            orig_vis = denorm(img_tensor)
            recon_vis = denorm(recon_img)
            diff_vis = diff

            combined = torch.cat([orig_vis, recon_vis, diff_vis], dim=3)

            save_path = os.path.join(output_dir, f"result_{i:02d}.png")
            utils.save_image(combined, save_path)


if __name__ == "__main__":
    test()