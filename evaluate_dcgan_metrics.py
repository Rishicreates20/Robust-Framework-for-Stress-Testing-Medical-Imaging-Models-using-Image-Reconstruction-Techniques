import os
import numpy as np
import torch
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from pytorch_fid import fid_score
from math import log10
import argparse
from tqdm import tqdm

# Function to calculate PSNR
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * log10(255.0 / np.sqrt(mse))

# Function to calculate IS (Inception Score)
def calculate_inception_score(fake_imgs, num_splits=10):
    # Assuming pre-trained InceptionV3 available (modify if required)
    from torchvision.models import inception_v3
    inception = inception_v3(pretrained=True, transform_input=False).eval()

    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    processed_imgs = torch.stack([preprocess(img) for img in fake_imgs]).cuda()
    with torch.no_grad():
        preds = inception(processed_imgs).softmax(dim=-1)
    scores = []
    for i in range(num_splits):
        part = preds[i * len(preds) // num_splits: (i + 1) * len(preds) // num_splits]
        py = torch.mean(part, dim=0)
        scores.append(torch.exp(torch.sum(py * torch.log(py + 1e-6))))
    return torch.mean(torch.tensor(scores)).item()

# Main evaluation function
def evaluate_metrics(real_dir, fake_dir):
    transform = transforms.ToTensor()

    # Load real and fake images
    real_imgs = [transform(Image.open(os.path.join(real_dir, img)).convert("RGB")) for img in os.listdir(real_dir)]
    fake_imgs = [transform(Image.open(os.path.join(fake_dir, img)).convert("RGB")) for img in os.listdir(fake_dir)]

    # Ensure equal number of images
    assert len(real_imgs) == len(fake_imgs), "Mismatch in the number of real and fake images!"

    # Convert to NumPy arrays
    real_imgs_np = [np.array(img.permute(1, 2, 0)) for img in real_imgs]
    fake_imgs_np = [np.array(img.permute(1, 2, 0)) for img in fake_imgs]

    # SSIM
    ssim_scores = [ssim(real, fake, multichannel=True) for real, fake in zip(real_imgs_np, fake_imgs_np)]
    avg_ssim = np.mean(ssim_scores)

    # FID
    fid = fid_score.calculate_fid_given_paths([real_dir, fake_dir], batch_size=50, device='cuda', dims=2048)

    # PSNR
    psnr_scores = [calculate_psnr(real, fake) for real, fake in zip(real_imgs_np, fake_imgs_np)]
    avg_psnr = np.mean(psnr_scores)

    # Inception Score (IS)
    avg_is = calculate_inception_score(fake_imgs)

    return avg_ssim, fid, avg_psnr, avg_is

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DCGAN Outputs using Metrics")
    parser.add_argument("--real_dir", type=str, required=True, help="Path to the directory containing real images")
    parser.add_argument("--fake_dir", type=str, required=True, help="Path to the directory containing fake images")
    args = parser.parse_args()

    print("Evaluating Metrics...")
    avg_ssim, fid, avg_psnr, avg_is = evaluate_metrics(args.real_dir, args.fake_dir)

    print(f"SSIM: {avg_ssim:.4f}")
    print(f"FID: {fid:.4f}")
    print(f"PSNR: {avg_psnr:.4f} dB")
    print(f"Inception Score (IS): {avg_is:.4f}")

