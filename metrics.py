import lpips
from math import isclose
import torch
import numpy as np
from skimage.metrics import structural_similarity
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip

# loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()

# as_model, as_preprocessor = convert_v2_5_from_siglip(
#     low_cpu_mem_usage=True,
#     trust_remote_code=True,
# )
# as_model = as_model.to(torch.bfloat16).cuda()

def convert_to_float32(x):
    x_float = (x / 255.0).astype(np.float32)  # Convert to float32 and normalize to [0, 1]
    return x_float

def PSNR(y1, y2):
    if y1.dtype == np.uint8:
        y1 = convert_to_float32(y1)
    if y2.dtype == np.uint8:
        y2 = convert_to_float32(y2)
    mse = np.mean((y1 - y2) ** 2)
    return 20 * np.log10(y1.max() / np.sqrt(mse))

def LPIPS(y1, y2):
    y1_n = preprocess_for_lpips(y1)
    y2_n = preprocess_for_lpips(y2)
    return loss_fn_vgg(y1_n.to("cuda"), y2_n.to("cuda")).mean().item()

def SSIM(y1, y2):
    # y1 = (y1 - y1.min()) / (y1.max() - y1.min()) # [0, 1]
    # y2 = (y2 - y2.min()) / (y2.max() - y2.min()) # [0, 1]
    if y1.dtype == np.uint8:
        y1 = convert_to_float32(y1)
    if y2.dtype == np.uint8:
        y2 = convert_to_float32(y2)
    ssim_list = [structural_similarity(y1[i], y2[i], data_range=1.0, channel_axis=2) for i in range(y1.shape[0])]
    return np.mean(np.array(ssim_list))

def preprocess_for_lpips(y):
    if y.shape[3] == 3:
        y = y.permute((0, 3, 1, 2)) # now [b, c, h, w]
    y_n = (y - y.min()) / (y.max() - y.min()) # [0, 1]
    return (y_n * 2.0) - 1.0 # [-1, 1]

def measurements_consistency(x, measurements, forward_op):
    if x.dtype == np.uint8:
        x = convert_to_float32(x)
    if measurements.dtype == np.uint8:
        measurements = convert_to_float32(measurements)
    forward_x = convert_to_float32(np.asarray(forward_op(x)))
    mse = np.mean((forward_x - measurements) ** 2)
    return mse

def aesthetic_score(x):
    # preprocess image
    pixel_values = (
        as_preprocessor(images=x, return_tensors="pt")
        .pixel_values.to(torch.bfloat16)
        .cuda()
    )

    # predict aesthetic score
    with torch.inference_mode():
        score = as_model(pixel_values).logits.squeeze().float().cpu().numpy()
    
    return score.mean()

# def InceptionScore(x):
#     if x.shape[3] == 3:
#         x = x.permute((0, 3, 1, 2)) # now [b, c, h, w]
#     x = (x - x.min()) / (x.max() - x.min()) # [0, 1]
#     IS, IS_std = get_inception_score(x, splits=1)
#     return IS

