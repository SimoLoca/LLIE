import os
import cv2
import numpy as np
import torch
import argparse
from methods import (
    DCE,
    enhance
)
import time
from skimage.metrics import (
    structural_similarity, 
    peak_signal_noise_ratio,
    mean_squared_error
)


def parse():
    parser = argparse.ArgumentParser(
      description='Evaluate different methods for low-light image enhancement on LOL dataset.')
    parser.add_argument('--image', type=str, required=True, 
                        help='Name of the image to evaluate.')
    return parser.parse_args()


def naive_gamma(image: np.ndarray, gamma: float = 1.5):
    # Apply gamma correction to enhance image brightness (gamma > 1)
    enhanced_image = np.power(image / 255.0, gamma) * 255.0
    enhanced_image = np.uint8(enhanced_image)
    return enhanced_image


def compute_ssim(img1: np.ndarray, img2: np.ndarray):
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return structural_similarity(gray_img1, gray_img2)


if __name__ == "__main__":
    args = parse()

    dataset_path = "../LOLdataset/our485/"
    low_light_image_path = os.path.join(dataset_path, "low", args.image)
    high_light_image_path = os.path.join(dataset_path, "high", args.image)

    assert os.path.exists(low_light_image_path), "Low-light image not found"
    assert os.path.exists(high_light_image_path), "High-light image not found"

    # Load the low-light image
    input_image = cv2.imread(low_light_image_path)
    h, w, c = input_image.shape

    assert input_image is not None, "Error: Could not load image"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load DCE model
    dce = DCE("./methods/DCE/ckpt/best_model.pth", device)

    start_time = time.time()
    naive_enh_image = naive_gamma(input_image, gamma=.5)
    end_time = time.time()
    naive_latency_ms = (end_time - start_time) * 1000

    enhance(input_image)    # warmp-up
    start_time = time.time()
    ivl_enh_image = enhance(input_image)
    end_time = time.time()
    ivl_latency_ms = (end_time - start_time) * 1000

    start_time = time.time()
    dce_enh_image = dce.enhance(low_light_image_path, size=(h, w))
    end_time = time.time()
    dce_latency_ms = (end_time - start_time) * 1000

    # Not working :(
    #cv2.imshow('Original Image', cv2.cvtColor(input_image, ))
    #cv2.imshow('Naive Enhanced Image', naive_enh_image)
    #cv2.imshow('IVL Enhanced Image', ivl_enh_image)
    #cv2.imshow('DCE Enhanced Image', dce_enh_image)

    # Save the enhanced images
    img_name = args.image.split('/')[-1].split('.')[0]
    if not os.path.isdir("./results/"):
        os.makedirs("./results/")
    cv2.imwrite(f'./results/{img_name}_naive_enhanced_image.jpg', naive_enh_image)
    cv2.imwrite(f'./results/{img_name}_ivl_enhanced_image.jpg', ivl_enh_image)
    cv2.imwrite(f'./results/{img_name}_dce_enhanced_image.jpg', dce_enh_image)

    ### Image Quality Assessment
    hl_image = cv2.imread(high_light_image_path, cv2.IMREAD_COLOR)

    psnr_naive = peak_signal_noise_ratio(hl_image, naive_enh_image)
    psnr_ivl = peak_signal_noise_ratio(hl_image, ivl_enh_image)
    psnr_dce = peak_signal_noise_ratio(hl_image, dce_enh_image)

    mse_naive = mean_squared_error(hl_image, naive_enh_image)
    mse_ivl = mean_squared_error(hl_image, ivl_enh_image)
    mse_dce = mean_squared_error(hl_image, dce_enh_image)

    mae_naive = np.mean(np.abs(hl_image.copy() - naive_enh_image.copy()))
    mae_ivl = np.mean(np.abs(hl_image.copy(), ivl_enh_image.copy()))
    mae_dce = np.mean(np.abs(hl_image.copy(), dce_enh_image.copy()))

    ssim_naive = compute_ssim(hl_image, naive_enh_image)
    ssim_ivl = compute_ssim(hl_image, ivl_enh_image)
    ssim_dce = compute_ssim(hl_image, dce_enh_image)


    print("+----------+---------------+----------+----------+----------+----------+")
    print(f"|{'Method':^10}|{'Latency (ms) ↓':^15}|{'PSNR ↑':^10}|{'MSE ↓':^10}|{'MAE ↓':^10}|{'SSIM ↑':^10}|")
    print("+----------+---------------+----------+----------+----------+----------+")
    print(f"|{'Naive':<10}|{naive_latency_ms:^15.3f}|{psnr_naive:^10.2f}|{mse_naive:^10.2f}|{mae_naive:^10.2f}|{ssim_naive:^10.2f}|")
    print(f"|{'IVL':<10}|{ivl_latency_ms:^15.3f}|{psnr_ivl:^10.2f}|{mse_ivl:^10.2f}|{mae_ivl:^10.2f}|{ssim_ivl:^10.2f}|")
    print(f"|{'DCE':<10}|{dce_latency_ms:^15.3f}|{psnr_dce:^10.2f}|{mse_dce:^10.2f}|{mae_dce:^10.2f}|{ssim_dce:^10.2f}|")
    print("+----------+---------------+----------+----------+----------+----------+")
