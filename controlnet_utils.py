import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
import torch
import random

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


def get_random_mask_for_inpainting(images, mask_size=(64, 64), num_masks=1, mask_value=255):
    """
    Generates random masks for inpainting tasks. The masks are random in shape and location, 
    suitable for use in inpainting models.
    
    Args:
    - images: List of PIL images (or a single image) to apply the random mask to.
    - mask_size: The size of the random mask (height, width).
    - num_masks: The number of random masks to generate per image.
    - mask_value: The value of the mask (usually 255 for a white mask).
    
    Returns:
    - masks: List of masks for each image. Each mask is a binary array with random regions.
    """
    # Ensure input is a list of PIL Images
    if isinstance(images, np.ndarray):
        if images.dtype != np.uint8:
            assert images.min() >= 0 and images.max() <= 1, "Float images should be in range [0, 1]"
            images = (images * 255).astype(np.uint8)
        images = [Image.fromarray(img) for img in images]
    elif isinstance(images, Image.Image):
        images = [images]

    masks = []

    for image in images:
        width, height = image.size
        mask = np.zeros((height, width), dtype=np.uint8)

        for _ in range(num_masks):
            # Randomly choose the position for the mask
            top_left_x = random.randint(0, width - mask_size[1])
            top_left_y = random.randint(0, height - mask_size[0])

            # Create a rectangular mask
            mask[top_left_y:top_left_y + mask_size[0], top_left_x:top_left_x + mask_size[1]] = mask_value

        masks.append(Image.fromarray(mask))

    return masks
