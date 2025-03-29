import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

def get_canny_image(image):
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image


def get_depth_estimation(image):
    checkpoint = "depth-anything/Depth-Anything-V2-base-hf"
    pipe = pipeline("depth-estimation", model=checkpoint, device="cuda")
    predictions = pipe(image)
    depth = predictions["depth"]
    depth = np.array(depth)
    depth = depth[:, :, None]
    depth = np.concatenate([depth, depth, depth], axis=2)
    depth = Image.fromarray(depth)
    return depth
