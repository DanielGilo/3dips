import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
import torch

def get_canny_image(images):
    images = np.array(images)
    canny_images = []
    if images.dtype != np.uint8:
        assert images.min() >= 0 and images.max() <= 1, "Float images should be in range [0, 1]"
        images = (images * 255).astype(np.uint8)
    for image in images:
        canny = cv2.Canny(image, 100, 200)
        canny = canny[:, :, None]
        canny = np.concatenate([canny, canny, canny], axis=2)
        canny_image = Image.fromarray(canny)
        canny_images.append(canny_image)
    return canny_images
    


def get_depth_estimation(images):
    if isinstance(images, np.ndarray):
        if images.dtype != np.uint8:
            assert images.min() >= 0 and images.max() <= 1, "Float images should be in range [0, 1]"
            images = (images * 255).astype(np.uint8)
        images = [Image.fromarray(img) for img in images]
    checkpoint = "depth-anything/Depth-Anything-V2-base-hf"
    pipe = pipeline("depth-estimation", model=checkpoint, device="cuda")
    predictions = pipe(images)
    pipe.model.to("cpu")
    depth = [pred["depth"] for pred in predictions]
    depth = np.array(depth)
    depth = depth[:, :, :, None]
    depth = np.concatenate([depth, depth, depth], axis=3) # [b, H, W, 3]
    depth_pil = [Image.fromarray(d) for d in depth]
    return depth_pil
    # depth = Image.fromarray(depth)
    # return depth


def get_mask_from_image_by_prompt(images, label):
    """
    Masks regions in the input image using Facebook's Segment Anything 2 (SAM2).
    Mask pixels are set to white. Mask is determined according to given prompt.
    """
    # Ensure input is a list of PIL Images
    if isinstance(images, np.ndarray):
        if images.dtype != np.uint8:
            assert images.min() >= 0 and images.max() <= 1, "Float images should be in range [0, 1]"
            images = (images * 255).astype(np.uint8)
        images = [Image.fromarray(img) for img in images]
    elif isinstance(images, Image.Image):
        images = [images]

    # Load the SAM2 pipeline once
    pipe = pipeline("image-segmentation", "nvidia/segformer-b1-finetuned-cityscapes-1024-1024", device="cuda")

    masks = []
    for image in images:
        results = pipe(image)
        mask = None
        for obj in results:
            if obj['label'] != label:
                continue
            masks.append(obj['mask'])
            break
        
    assert len(masks) == len(images), "Not all images have the specified label."

    pipe = None  # Free up memory
    torch.cuda.empty_cache()

    return masks