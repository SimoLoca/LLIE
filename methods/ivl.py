"""
Implementation of: "Shallow camera pipeline for night photography rendering"
Here we are using Non-Local Means, instead of BM3D, in the denoising stage.
Also, "Preliminary steps" are not performed.
"""

import cv2
import numpy as np


def convert2uint8(image: np.ndarray):
    return np.clip((image * 255.0), 0, 255).astype(np.uint8)


def lcc(image: np.ndarray):
    """
    Local Color Correction Using Non-Linear Masking Modded
    """
    ### Convert to YCbCr (YCrCb in open-cv)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ### Compute the inverted low-pass filtered mask
    # Take intensity channel Y
    y = ycrcb[:, :, 0]
    # Invert it: tells which region will be lightened or dakened
    y_inverted = 255 - y
    # Compute a blurred mask of the inverted channel and rescale
    y_blurred = cv2.GaussianBlur(y_inverted / 255.0, (3, 3), sigmaX=0.5) * 255
    # Compute the exp of the mask
    y_mean = np.mean(y)
    lambd = np.log(128)/np.log(y_mean) if y_mean >= 128 else np.log(y_mean)/np.log(128)
    exp = lambd**((128-(255-y_blurred))/128)
    # New Y channel: pixel-wise gamma correction
    new_y = 255*(y/255)**exp
    # Replace Y channel
    ycrcb[:, :, 0] = new_y
    # Convert to BGR
    bgr_result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return bgr_result, ycrcb, y


def contrast_enhancement(y_before: np.ndarray, ycrcb_after: np.ndarray):
    """
    Adaptively stretches and clips the image histogram based on how the
    distribution of dark pixels changes before and after the contrast 
    correction of LCC.
    """
    def dark_pixels(ycrcb: np.ndarray):
        """"
        Find dark pixels in an YCrCb image.
        """
        y = ycrcb[:, :, 0]
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        chroma_radius = ((cb-128)*2 + (cr-128)*2)/2
        darkPixels_idx = (y < 35) & (chroma_radius < 20)
        num_darkPixels = np.count_nonzero(darkPixels_idx)
        # print(f"num_darkPixels: {num_darkPixels}/{y.shape[0]*y.shape[1]} -> {(num_darkPixels/(y.shape[0]*y.shape[1]))*100}")
        found = True if num_darkPixels > 0 else False
        return found, num_darkPixels
    
    is_dark, num_dark_pixels = dark_pixels(ycrcb_after)

    y_after_Hist, _ = np.histogram(ycrcb_after[:, :, 0], 256, (0, 255))
    y_before_Hist, _ = np.histogram(y_before, 256, (0, 255))

    # Find lower and upper range for histogram stretching
    lower_range = 0
    upper_range = np.percentile(y_after_Hist, 98)
    if is_dark:
        y_after_cumulative_hist = np.cumsum(y_after_Hist)
        y_before_cumulative_hist = np.cumsum(y_before_Hist)
        # Predefined threshold
        threshold = num_dark_pixels * 0.3
        # print("threshold", threshold)
        # Compute the bin corresponding to the 30% dark pixels in the cumulative histograms
        if len(y_after_cumulative_hist[y_after_cumulative_hist <= threshold]) < 1 or \
            len(y_before_cumulative_hist[y_before_cumulative_hist <= threshold]) < 1:
            lower_range = np.percentile(y_after_Hist, 2)
        else:
            b_input30 = y_before_cumulative_hist[y_before_cumulative_hist <= threshold][-1]
            b_output30 = y_after_cumulative_hist[y_after_cumulative_hist <= threshold][-1]
            # print(b_output30, b_input30)
            lower_range = float(b_output30 - b_input30)
    else:
        lower_range = np.percentile(y_after_Hist, 2)
    
    max_bins_to_clip = 50
    lower_range = min(lower_range, max_bins_to_clip)
    upper_range = min(upper_range, 256 - max_bins_to_clip)

    # Apply histogram stretching and clipping
    ycrcb_after[:, :, 0] = np.clip(
        (ycrcb_after[:, :, 0] - lower_range) / (upper_range - lower_range) * 255,
        0,
        255,
    ).astype(np.uint8)

    return ycrcb_after

def saturation_enhancement(image: np.ndarray, y_before: np.ndarray, y_after: np.ndarray):
    """
    Saturation enhancement as in: "Adaptive gamma processing of the video 
    cameras for the expansion of the dynamic range".
    """
    bgr = cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR) / 255.
    y_before = y_before / 255.
    y_after = y_after / 255.
    for ch_idx in range(bgr.shape[-1]):
        idx = np.where(y_before == 0.0)
        y_before[idx] = 0.001
        bgr[:, :, ch_idx] = 0.5 * (y_after/y_before) * (bgr[:, :, ch_idx]+y_before) + bgr[:, :, ch_idx] - y_before
    return convert2uint8(bgr)

def cs_enhancement(y_before: np.ndarray, ycrcb_after: np.ndarray):
    ### First, apply contrast enhancement
    contr_image = contrast_enhancement(y_before, ycrcb_after)
    # cv2.imshow('2.1 - Image after contrast', cv2.cvtColor(contr_image, cv2.COLOR_YCrCb2BGR))
    ### Then, apply saturation enhancement
    cs_image = saturation_enhancement(contr_image, y_before, ycrcb_after[:, :, 0])
    return cs_image


def black_point_gamma_correction(image: np.ndarray):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ### Black point correction
    v = hsv[:, :, 2]
    #print("V", v)
    v_percentile = np.percentile(v, 2) # 20-th percentile in the original paper
    #print("v_percentile", v_percentile)
    v[v < v_percentile] = 0.
    hsv[:, :, 2] = v

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    ### Gamma correction
    gamma = 1/1.4
    #bgr = np.power(bgr / 255.0, gamma)
    lut = np.empty((1, 256), np.uint8)
    for i in range(256):
        lut[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    bgr = cv2.LUT(bgr, lut)/255.
    return bgr


def unsharp_masking(image: np.ndarray):
    # Compute low freq. mask
    blurred = cv2.GaussianBlur(image, (5, 5), 2.5)
    mask = image - blurred
    mask_strength = 1.25
    # Compute unsharp masking
    sharpened_image = image + mask_strength * mask
    return sharpened_image


def denoising(image: np.ndarray, original_image: np.ndarray):
    # Convert the image to YCbCr color space
    ycbcr_original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2YCrCb)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    # Extract the Y channel
    y_original_image = ycbcr_original_image[:, :, 0]
    y_image = image[:, :, 0]

    # Apply Non-Local Means denoising to the Y channel, instead of BM3D
    filtered = cv2.fastNlMeansDenoising(y_image, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # Define the u parameter (controls denoising effect in bright areas)
    u = 0.6
    # Generate the mask by blurring the luminance channel Y with a Gaussian filter
    mask = cv2.GaussianBlur(y_original_image / 255.0, (5, 5), 1)

    # Compute the denoised image D
    denoised_image = filtered * (1 - mask * u) + y_original_image * (mask * u)

    # Convert the Y channel back to 3 channels
    image[:, :, 0] = denoised_image

    # Convert the image back to BGR color space
    denoised_bgr_image = cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)
    return denoised_bgr_image


def white_balance(image: np.ndarray):
    result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


def show_enhance(image):
    cv2.imshow('0 - Low-Light Image', image)

    img1, ycrcb_after, y_before = lcc(image)
    cv2.imshow('1 - Image after LCC', img1)

    img2 = cs_enhancement(y_before, ycrcb_after)
    cv2.imshow('2 - Image after Contrast & Saturation Enhancement', img2)

    img3 = black_point_gamma_correction(img2)
    cv2.imshow('3 - Image after Black point & Gamma-correction', img3)

    img4 = unsharp_masking(img3)
    cv2.imshow('4 - Image after Unsharp-Masking', img4)

    image_8bit = convert2uint8(img4)
    cv2.imshow('5 - Image after 8-bit encoding', image_8bit)

    denoised_image = denoising(image_8bit, image)
    cv2.imshow('6 - Image after Denoising', denoised_image)

    result = white_balance(denoised_image)
    cv2.imshow('7 - Final result', result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return result


def enhance(image: np.ndarray):
    """
    Shallow camera pipeline for night photography rendering.
    """
    ### 1 - Local Contrast Correction (LCC)
    img1, ycrcb_after, ycrcb_before = lcc(image)
    ### 2 - Contrast & Saturation Enhancement (application of LCC tends to reduce the overall contrast and saturation)
    img2 = cs_enhancement(ycrcb_before, ycrcb_after)
    ### 3 - Black point & Gamma-correction
    img3 = black_point_gamma_correction(img2)
    ### 4 - Sharpening
    img4 = unsharp_masking(img3)
    ### 5 - 8-bit Encoding
    image_8bit = convert2uint8(img4)
    ### 6 - Denoising
    denoised_image = denoising(image_8bit, image)
    ### 7 - White balance
    result = white_balance(denoised_image)
    return result


def test(image):
    show_enhance(image)

if __name__ == "__main__":
    image = cv2.imread("/home/user/Desktop/LOLdataset/our485/low/504.png")
    test(image)