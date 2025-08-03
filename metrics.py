import lpips
from math import isclose
import torch
import numpy as np
from skimage.metrics import structural_similarity
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
from torchmetrics.multimodal import CLIPImageQualityAssessment
import clip
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from met3r import MEt3R

# loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()

as_model, as_preprocessor = convert_v2_5_from_siglip(
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
as_model = as_model.to(torch.bfloat16).cuda()

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
    forward_x = np.asarray(forward_op(x))
    if forward_x.dtype == np.uint8:
        forward_x = convert_to_float32(forward_x)
    assert forward_x.min() >= 0 and forward_x.max() <= 1 and measurements.min() >= 0 and measurements.max() <= 1, "float images should be in [0, 1] range."
    
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

# https://lightning.ai/docs/torchmetrics/stable/multimodal/clip_iqa.html
def clip_quality_assessment(imgs):
    if imgs.dtype == np.uint8:
        imgs = convert_to_float32(imgs)
    imgs_pt = torch.tensor(imgs).permute((0, 3, 1, 2))
    metric = CLIPImageQualityAssessment(data_range=1.0, prompts=("quality", "sharpness", "real"))
    res_dict = metric(imgs_pt)
    # Take mean of each list of scores
    for k in res_dict:
        res_dict[k] = res_dict[k].mean().item()
    return res_dict

# def InceptionScore(x):
#     if x.shape[3] == 3:
#         x = x.permute((0, 3, 1, 2)) # now [b, c, h, w]
#     x = (x - x.min()) / (x.max() - x.min()) # [0, 1]
#     IS, IS_std = get_inception_score(x, splits=1)
#     return IS



### ClipSimilarity code is copied from IN2N code

class ClipSimilarity(nn.Module):
    def __init__(self, name: str = "ViT-L/14"):
        super().__init__()
        assert name in ("RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px")  # fmt: skip
        self.size = {"RN50x4": 288, "RN50x16": 384, "RN50x64": 448, "ViT-L/14@336px": 336}.get(name, 224)

        self.model, _ = clip.load(name, device="cpu", download_root="./")
        self.model.eval().requires_grad_(False)

        self.register_buffer("mean", torch.tensor((0.48145466, 0.4578275, 0.40821073)))
        self.register_buffer("std", torch.tensor((0.26862954, 0.26130258, 0.27577711)))

    def encode_text(self, text):
        text = clip.tokenize(text, truncate=True).to(next(self.parameters()).device)
        text_features = self.model.encode_text(text)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    def encode_image(self, image):  # Input images in range [0, 1].
        image = F.interpolate(image.float(), size=self.size, mode="bicubic", align_corners=False)
        image = image - rearrange(self.mean, "c -> 1 c 1 1")
        image = image / rearrange(self.std, "c -> 1 c 1 1")
        image_features = self.model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def forward(
        self, image_0, image_1, text_0, text_1
    ):
        image_features_0 = self.encode_image(image_0)
        image_features_1 = self.encode_image(image_1)
        text_features_0 = self.encode_text(text_0)
        text_features_1 = self.encode_text(text_1)
        sim_0 = F.cosine_similarity(image_features_0, text_features_0)
        sim_1 = F.cosine_similarity(image_features_1, text_features_1)
        sim_direction = F.cosine_similarity(image_features_1 - image_features_0, text_features_1 - text_features_0)
        sim_image = F.cosine_similarity(image_features_0, image_features_1)
        return sim_0, sim_1, sim_direction, sim_image


def clip_scores(orig_images, new_images, orig_prompt, new_prompt):
    # convert to torch float tensor 
    if orig_images.dtype == np.uint8:
        orig_images = convert_to_float32(orig_images)
    if new_images.dtype == np.uint8:
        new_images = convert_to_float32(new_images)
    if orig_images.shape[3] == 3:
        orig_images = orig_images.transpose((0, 3, 1, 2))  # now [b, c, h, w]
    if new_images.shape[3] == 3:
        new_images = new_images.transpose((0, 3, 1, 2))  # now [b, c, h, w]
    orig_images = torch.tensor(orig_images).float().cuda()
    new_images = torch.tensor(new_images).float().cuda()

    clip_similarity = ClipSimilarity().cuda()
    orig_images_features = clip_similarity.encode_image(orig_images)
    new_images_features = clip_similarity.encode_image(new_images)
    orig_prompt_features = clip_similarity.encode_text(orig_prompt)
    new_prompt_features = clip_similarity.encode_text(new_prompt) 

    clip_sim_score = F.cosine_similarity(new_images_features, new_prompt_features).mean().item()
    clip_dir_score = F.cosine_similarity(new_images_features - orig_images_features,
                                                new_prompt_features - orig_prompt_features).mean().item()

    # typo in formula in IN2N Appendix C: https://arxiv.org/pdf/2303.12789#page=6.89
    # author commented in issue: https://github.com/ayaanzhaque/instruct-nerf2nerf/issues/28
    # formula is (C(oi+1) - C(oi)) \dot (C(ei+1) - C(ei))
    orig_images_consequtive_diff = orig_images_features.diff(dim=0)
    new_images_consequtive_diff = new_images_features.diff(dim=0)
    clip_dir_consistency_score = F.cosine_similarity(new_images_consequtive_diff, orig_images_consequtive_diff).mean().item()

    return clip_sim_score, clip_dir_score, clip_dir_consistency_score

def compute_met3r_successive_frames(frames: torch.Tensor, batch_size = 5):
    """
    Compute the MEt3R metric for successive pairs of frames.

    Args:
        frames (torch.Tensor): Tensor of shape (N, 3, H, W).

    Returns:
        torch.Tensor: Mean MEt3R score for all pairs.
    """
    if frames.dtype == np.uint8:
        frames = convert_to_float32(frames) # frames in [0, 1] range

    # Convert to torch tensor if needed
    if isinstance(frames, np.ndarray):
        frames = torch.from_numpy(frames)
    
    if frames.ndim != 4:
        raise ValueError(f"Expected input with 4 dimensions, got {frames.ndim}")
    
    # Convert (N, H, W, 3) -> (N, 3, H, W) if needed
    if frames.shape[-1] == 3 and frames.shape[1] != 3:
        frames = frames.permute(0, 3, 1, 2)

    if frames.min() >=0:
        frames = frames * 2 - 1 # Normalize to [-1, 1]
    
    img_size = frames.shape[2]
    N = frames.shape[0]
    assert frames.shape[1] == 3, "Expected 3 channels in second dimension"
    assert N >= 2, "Need at least 2 frames for pairwise comparison"

    # Prepare input of shape (N-1, 2, 3, H, W)
    pair_inputs = torch.stack([frames[:-1], frames[1:]], dim=1).to("cuda")

    # Initialize MEt3R
    metric = MEt3R(
        img_size=img_size,  # or None for dynamic sizing
        use_norm=True,
        backbone="mast3r",
        feature_backbone="dino16",
        feature_backbone_weights="mhamilton723/FeatUp",
        upsampler="featup",
        distance="cosine",
        freeze=True,
    ).to("cuda")

    torch.use_deterministic_algorithms(False, warn_only=False) # Met3R requires non-deterministic algorithms for some operations

    # Compute scores
    all_scores = []
    with torch.no_grad():
        for i in range(0, pair_inputs.shape[0], batch_size):
            batch = pair_inputs[i:i + batch_size]
            scores, *_ = metric(
                images=batch,
                return_overlap_mask=False,
                return_score_map=False,
                return_projections=False
            )
            all_scores.append(scores)
    
    torch.cuda.empty_cache()

    torch.use_deterministic_algorithms(True, warn_only=False)

    # Concatenate and compute mean
    all_scores = torch.cat(all_scores, dim=0)
    assert len(all_scores) == len(frames) - 1, "Scores length should be N-1"
    return all_scores.mean().item()



def run_metric_suite_controlled(output_frames, measurements, forward_op, wb=None):
    meter_score = compute_met3r_successive_frames(output_frames, batch_size=1)
    clip_iqa_dict = clip_quality_assessment(output_frames)
    aesthetic = aesthetic_score(output_frames)
    final_mc = measurements_consistency(output_frames, np.asarray(measurements), forward_op)

    res_dict = {"clip-quality": clip_iqa_dict["quality"],
            "clip-sharpness": clip_iqa_dict["sharpness"],
            "clip-realism": clip_iqa_dict["real"],
             "final-measurements_consistency": final_mc,
             "meter-score": meter_score,
             "aesthetic_score": aesthetic}

    if wb is not None:
        wb.log(res_dict)
    
    print(res_dict)
    

def run_metric_suite_editing(original_frames, output_frames, original_prompt, edited_prompt, wb=None):
    meter_score = compute_met3r_successive_frames(output_frames, batch_size=1)
    clip_iqa_dict = clip_quality_assessment(output_frames)
    aesthetic = aesthetic_score(output_frames)

    clip_sim_score, clip_dir_score, clip_dir_consistency_score = clip_scores(
                                                                    orig_images=original_frames,
                                                                    new_images=output_frames,
                                                                    orig_prompt=original_prompt,
                                                                    new_prompt=edited_prompt)

    res_dict = {"clip-quality": clip_iqa_dict["quality"],
            "clip-sharpness": clip_iqa_dict["sharpness"],
            "clip-realism": clip_iqa_dict["real"],
            "meter-score": meter_score,
            "aesthetic_score": aesthetic,
            "clip-sim-score": clip_sim_score,
            "clip_dir_score": clip_dir_score,
            "clip_dir_consistency_score": clip_dir_consistency_score}

    if wb is not None:
        wb.log(res_dict)
    
    print(res_dict)