import argparse
import os
import shutil
import copy
from os.path import basename
import torch
import torch.nn.functional as F
#torch.autograd.set_detect_anomaly(True)

import gc
import numpy as np
import random
from scipy.stats import truncnorm
import wandb
from torch.optim.sgd import SGD
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchvision.transforms.functional import to_pil_image, pil_to_tensor


from teachers import get_teacher
from students import get_mv_student
from losses import get_loss_f
from metrics import PSNR, measurements_consistency, SSIM, run_metric_suite_controlled, run_metric_suite_editing
from seva_utils import get_value_dict_of_scene
from plot_utils import plot_frames_row, plot_logs, plot_frames
from latent_translator import LatentTranslator
from diffusion_pipeline import SevaPipeline

from seva.seva.sampling import to_d, append_dims


device = torch.device('cuda:0')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # avoids memory fragmentation
os.environ["CUBLAS_WORKSPACE_CONFIG"] =":4096:8" # necessary when setting torch.use_dererministic_algorithms(True), increases memory by 24MB

rng = np.random.default_rng(42)

def seed_everything(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)
    print(f"Seed set to {seed}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse user arguments for an experiment.")

    parser.add_argument("--teacher_name", type=str, required=True, help="Name of the Teacher (string).")
    parser.add_argument("--scene_path", type=str, required=True, help="Path to the scene folder")
    parser.add_argument("--num_views", type=int, required=True, help="Number of total views (input + output)")
    parser.add_argument("--prompt", type=str, required=True, help="Original prompt describing the scene")
    parser.add_argument("--editing_prompt", type=str, default="", help="Instructional editing prompt (for editing teachers)")
    parser.add_argument("--edited_prompt", type=str, default="", help="Final appearance prompt after editing")
    parser.add_argument("--student_name", type=str, required=True, help="Name of the Student (string).")
    parser.add_argument("--loss", type=str, required=True, help="Name of the loss function to use (string).")
    parser.add_argument("--lr", type=float, required=True, help="Max learning rate (float).")
    parser.add_argument("--final_lr", type=float, required=True, help="Final learning rate (float).")
    parser.add_argument("--n_warmup_steps", type=int, default=400, help="Number of warmup steps (integer, [0, inf)).")
    parser.add_argument("--n_distill_per_timestep", type=int, default=400, help="Number of distillation iterations per timestep (integer).")
    parser.add_argument("--n_distill_initial_timestep", type=int, required=True, help="Number of distillation iterations of the first timestep (integer).")
    parser.add_argument("--distill_dt", type=int, default=25, help="Timestep gap for the distillation process (integer, [1, 999]).")
    parser.add_argument("--sample_dt", type=int, default=10, help="Timestep gap for the sampling process (integer, [1, 999]).")
    parser.add_argument("--n_inv_iters", type=int, default=0, help="Number of inversion iterations (integer).")
    parser.add_argument("--n_teacher_iters", type=int, default=1, help="Number of teacher DDIM iterations (integer).")
    parser.add_argument("--pred_teacher_interval", type=int, default=1, help="Number distillation iterations to use the same teacher prediction (integer).")
    parser.add_argument("--eta_teacher", type=float, default=0.0, help="Noise stochasticity parameter [0,1].")
    parser.add_argument("--t_min", type=int, default=0, help="Minimal timestep for distillation [0, 999] (integer).")
    parser.add_argument("--conv_thresh", type=float, default=1.0e-2, help="Relative loss convergence threshold for distillation of a timestep (float).")
    parser.add_argument("--cfg_w", type=float, default=2.0, help="Student CFG weight (float).")
    parser.add_argument("--teacher_cfg", type=float, default=2.0, help="Teacher CFG weight (float).")
    parser.add_argument("--image_cfg", type=float, default=1.5, help="Pix2pix Teacher image CFG weight (float).")
    parser.add_argument("--sampling_guidance_scale", type=float, default=0.0, help="Guidance scale for sampling stage (float).")
    parser.add_argument("--distillation_guidance_scale", type=float, default=1.0, help="Distillation guidance scale (float).")
    parser.add_argument("--do_compile", action='store_true', help="Boolean flag to compile models.")
    parser.add_argument("--teacher_timestep_shape_factor", type=float, default=0.5, help="The shape factor for the truncated normal distribution that the teacher timestep is sampled from.\n"
                                                                                    "A value of 0.5 would result in a roughly uniform distribution between student_t and T, and higher values" \
                                                                                    " would result in a lower variance and higher probability around the lower bound -- student_t (float).")

    args = parser.parse_args()
    return args


def check_for_early_stopping(losses, window, threshold):
    if len(losses) < window:
        return False  # Not enough data to check
    recent = np.array(losses[-window:])
    max_loss = np.abs(recent).max()
    min_loss = np.abs(recent).min()
    # Relative convergence: difference is small compared to max_loss
    if max_loss == 0:
        return True  # Already converged to zero
    return ((max_loss - min_loss) / max_loss) < threshold


def print_max_grad_value(params):
    max_grad = 0.0
    for p in params:
        if p.grad is not None:
            param_max = p.grad.abs().max().item()
            if param_max > max_grad:
                max_grad = param_max
    print(f"Max gradient value: {max_grad}")


def print_num_trained_params(optimizer):
    params = [p for group in optimizer.param_groups for p in group['params']]
    num_params = sum(p.numel() for p in params if p.requires_grad)
    print(f"Number of trainable parameters in optimizer: {num_params//1e6}M")


def get_n_distill_iters_for_t(t, min_t, n_distill_iters, n_distill_iters_initial_t):
    if t == 999:
        return n_distill_iters_initial_t
    if t >= min_t:
        return n_distill_iters
    return 0

def right_skewed_truncated_normal(mean, upper_bound, shape_factor=3.0, size=1000):
    if mean == upper_bound:
        return np.array([mean])
    # Set scale relative to the range
    scale = (upper_bound - mean) / shape_factor
    
    # Standardized bounds for truncnorm
    a, b = 0, (upper_bound - mean) / scale
    return truncnorm.rvs(a=a, b=b, loc=mean, scale=scale, size=size, random_state=rng).astype(int)

@torch.no_grad()
def log_visualizations(student, teacher, latent, student_z0s_teacher_space, z_ts, pred_z0s, wb, timestep, teacher_timestep):
    #out_frames = student.decode(z0_student_out_frames, type="np")
    out_frames = teacher.decode(torch.cat(student_z0s_teacher_space, dim=0).to(device="cuda", dtype=torch.float16), type="np")
    z_t_frames = teacher.decode(torch.cat(z_ts, dim=0).to(device="cuda", dtype=torch.float16), type="np")
    pred_z0_frames = teacher.decode(torch.cat(pred_z0s, dim=0).to(device="cuda", dtype=torch.float16), type="np")
    # latent_frames = student.decode(latent, type="np")


    plot_frames(out_frames, wb, "z", "t={}".format(timestep.item()))
    plot_frames(pred_z0_frames, wb, "teacher_pred", "t={}".format(teacher_timestep.item()))
    # plot_frames(z_t_frames, wb, "z_t", "t={}".format(teacher_timestep.item()))
    # plot_frames(teacher.forward_operator(out_frames), wb, "forward_op(z)", "t={}".format(timestep.item()))
    
    output_log["z"].append(out_frames)
    output_log["z_t"].append(z_t_frames)
    output_log["teacher_pred"].append(pred_z0_frames)
    output_log["t"].append(timestep.item())

    
    wb.log({#"z_video": wandb.Video((out_frames * 255).astype(np.uint8).transpose((0, 3, 1, 2)), format="mp4", fps=3, caption="z0(t={})".format(timestep.item())),
            "PSNR": PSNR(out_frames, np.asarray(teacher.gt_images)),
            "measurements_consistency": measurements_consistency(out_frames, np.asarray(measurments), teacher.forward_operator),
            "SSIM": SSIM(out_frames, np.asarray(teacher.gt_images)),
            "teacher_PSNR": PSNR(pred_z0_frames, np.asarray(teacher.gt_images)),
            "teacher_MC": measurements_consistency(pred_z0_frames, np.asarray(measurments), teacher.forward_operator),
            "teacher_SSIM": SSIM(pred_z0_frames, np.asarray(teacher.gt_images))})


    #wb.log({"LPIPS": LPIPS(torch.tensor(out_frames).permute((0, 3, 1, 2)), torch_gt)})
    #wb.log({"Aesthetic": aesthetic_score(out_frames)})
    
    #wb.log({"teacher_LPIPS": LPIPS(torch.tensor(pred_z0_frames).permute((0, 3, 1, 2)), torch_gt)})
    #wb.log({"teacher_Aesthetic": aesthetic_score(pred_z0_frames)})


seed_everything(seed=42)

args = parse_arguments()
teacher_name = args.teacher_name
lr = args.lr
n_distill_per_timestep = args.n_distill_per_timestep
student_name = args.student_name
warmup_steps = args.n_warmup_steps
text_target = args.prompt 
editing_prompt = args.editing_prompt
edited_scene_prompt = args.edited_prompt
loss_f = get_loss_f(args.loss)
cfg_w = args.cfg_w

t_max = 999
t_min_distill = args.t_min
n_distill_timesteps = ((1000 - t_min_distill) // args.distill_dt) + 1

timesteps = torch.linspace(t_max, t_min_distill, n_distill_timesteps, device="cuda", dtype=torch.int)
n_timesteps = len(timesteps)

kernel_size = 5

min_lr = 1e-9
max_lr = lr
final_lr = args.final_lr
#decay_steps = int(n_timesteps * n_distill_per_timestep - warmup_steps)
decay_steps = int((1000 / args.distill_dt) * n_distill_per_timestep - warmup_steps) # decay until t=0 even if t_min>0

def lr_lambda(step):
    if max_lr < 1e-14:
        return 0.0
    if step < warmup_steps:
        # Linear warmup: scale from base_lr to max_lr
        factor = (min_lr / max_lr) * (1 - (step / (warmup_steps -1))) + 1 * (step / (warmup_steps - 1))
        return factor
    elif step < warmup_steps + decay_steps:
        # Cosine decay: scale from max_lr to final_lr
        decay_step = step - warmup_steps
        cosine_decay = 0.5 * (1 + np.cos(np.pi * decay_step / decay_steps))
        return (final_lr + (max_lr - final_lr) * cosine_decay) / max_lr
    else:
        # Constant final_lr
        return final_lr / max_lr


scene_path = args.scene_path
exp_name = f"{student_name}-{teacher_name}-{basename(scene_path)[:10]}-cfg_teacher-{args.teacher_cfg}-cfg_image-{args.image_cfg}-shape-f-{args.teacher_timestep_shape_factor}-seva-1.1"
#exp_name = "test-determinism"

config = {"text_target": text_target,  "n_timesteps": n_distill_timesteps, "n_distill_per_t": n_distill_per_timestep,
            "lr": lr, "teacher_name": teacher_name, "student_name":student_name, "loss": args.loss, "scene": scene_path,
            "t_min_distillation": t_min_distill, "editing_prompt": editing_prompt}
wb = wandb.init(
    project="3dips",
    name=exp_name,
    config=config
);
wb.log_code()


num_inputs = 1
num_views_total = args.num_views
input_json = os.path.join(scene_path, f"{num_views_total}_train_test_split_1.json")
output_json = os.path.join(scene_path, "train_test_split_1.json")
shutil.copyfile(input_json, output_json)

value_dict = get_value_dict_of_scene(scene_path, num_inputs)
value_dict_teacher = {} # get_value_dict_of_scene(scene_path, 4)

all_gt = value_dict["cond_frames"]
torch_gt = (((all_gt[num_inputs:value_dict["num_imgs_no_padding"]] + 1) / 2.0)* 255).clamp(0, 255).to(torch.uint8)
pil_gt = [to_pil_image(torch_gt[i]) for i in range(torch_gt.shape[0])]


num_iters = n_distill_per_timestep * n_distill_timesteps
show_interval = 1 #int(num_iters / min(1, num_iters))

student = get_mv_student(student_name, value_dict=value_dict, device=device, latent_shape=[21, 4, 72, 72], do_compile=args.do_compile)
teacher = get_teacher(teacher_name, prompt=text_target, editing_prompt=editing_prompt, image_guidance_scale=args.image_cfg,
                                value_dict=value_dict_teacher, device=device, gt_images=pil_gt, do_compile=args.do_compile)
teacher.plot_input(wb)
teacher_res = (teacher.pixel_space_shape[1], teacher.pixel_space_shape[2])  # (H, W)

saved_model_weights = copy.deepcopy(student.model.state_dict())

latent_translator = LatentTranslator(
    in_channels=student.latent_shape[1], out_channels=teacher.latent_shape[0],
    in_size=student.latent_shape[2:], out_size=teacher.latent_shape[1:], kernel_size=kernel_size,
    init_identity=True
).to(device)


output_log = {"z": [], "z_t": [], "teacher_pred": [], "t": []}

measurments = teacher.forward_operator(teacher.gt_images)

latent_shape = student.latent_shape
latent = torch.randn(latent_shape).to("cuda")
latent *= torch.sqrt(1.0 + student.discretization(1000, device="cuda")[0] ** 2.0) # from seva.sampling prepare_sampling_loop() - not sure why necessary
s_in = latent.new_ones([latent.shape[0]])

    
scaler = torch.cuda.amp.GradScaler()
optimizer = AdamW(params=list(student.get_trainable_parameters()) + list(latent_translator.parameters()),
                 lr=lr, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

print_num_trained_params(optimizer)

bsz = len(pil_gt) #20
assert len(pil_gt) % bsz == 0, "num test frames must be divisible by batch size"
n_batches = len(pil_gt) // bsz

iter_count = 0
convergence_threshold = args.conv_thresh  # Threshold for loss change to consider convergence
convergence_patience = 10  # Number of iterations to check for convergence

for time_i in range(n_timesteps):
    student_t = timesteps[time_i] * s_in # replicate for N_frames
    
    loss_history = []  

    n_distill_iters = get_n_distill_iters_for_t(timesteps[time_i], t_min_distill, n_distill_per_timestep, args.n_distill_initial_timestep)
    pred_z0 = None  # Initialize pred_z0 to None for the first iteration

    if time_i < n_timesteps - 1:
        next_t = timesteps[time_i+1] * s_in
        # with torch.no_grad():
        #     curr_weights  = student.model.state_dict()
        #     student.model.load_state_dict(saved_model_weights)
        #     undistilled_latent, _ = student.euler_edm_step(latent, student_t, next_t, cfg_w)
        #     student.model.load_state_dict(curr_weights)
    # if time_i > 0:
    #     student.model.load_state_dict(saved_model_weights)

    for distill_i in range(n_distill_iters):
        gc.collect()
        torch.cuda.empty_cache()
        optimizer.zero_grad()

        
        pred_eps, z0_student = student.predict_eps_and_sample(latent, student_t, cfg_w)
        assert torch.isfinite(z0_student).all()
        z0_student_out_frames = student.get_only_output_frames(z0_student)

        # eps = torch.randn_like(pred_eps, device="cuda")
        pred_eps_out_frames = student.get_only_output_frames(pred_eps).detach()

        timestep = torch.stack([torch.tensor(student_t[0].item(), dtype=torch.int)], dim=0).to("cuda")

        #w_t = student.get_sigma_from_timestep(timestep).item() ** 2
        w_t = 1

        if distill_i % args.pred_teacher_interval == 0:
            target_latent = None
        else:
            target_latent = pred_z0

        #teacher_timestep = timestep #- 30
        #teacher_timestep = torch.randint(low=t_min, high=t_max+1, size=(1,), device="cuda")
        #teacher_timestep = torch.randint(low=timestep.item(), high=t_max+1, size=(1,), device="cuda")
        teacher_timestep = right_skewed_truncated_normal(timestep.item(), upper_bound=t_max, 
                            shape_factor=args.teacher_timestep_shape_factor, size=1)[0] * torch.ones_like(timestep)

        z_ts = []
        pred_z0s = []
        student_z0s_teacher_space = []
        for batch_i in range(n_batches):
            batch_start = batch_i * bsz
            batch_end = batch_start + bsz

            if args.loss == "sds_shared": # shared latent space
                student_pred = latent_translator(z0_student_out_frames[batch_start:batch_end]).to(teacher.dtype)
                eps = torch.randn_like(student_pred, device="cuda")
                #eps = latent_translator(pred_eps_out_frames[batch_start:batch_end]).to(teacher.dtype)
            else:
                z0_student_out_frames_dec = student.decode(z0_student_out_frames[batch_start:batch_end], type="pt")
                student_pred =  F.interpolate(z0_student_out_frames_dec, size=teacher_res, mode='bilinear', align_corners=False)
                eps = torch.randn((bsz, 4, teacher_res[0] // 8, teacher_res[1] // 8), device="cuda")

            loss, z_t, pred_z0, student_z0_teacher_space = loss_f(
                            student_pred, 
                            teacher, 
                            args.teacher_cfg, 
                            eps,
                            target_latent, 
                            teacher_timestep, 
                            w_t,
                            args.eta_teacher,
                            batch_start,
                            batch_end,
                            args.n_inv_iters,
                            args.n_teacher_iters)
            
            z_ts.append(z_t.clone().detach().cpu())
            pred_z0s.append(pred_z0.clone().detach().cpu())
            student_z0s_teacher_space.append(student_z0_teacher_space.clone().detach().cpu())

            del z_t

            gc.collect()
            torch.cuda.empty_cache()

            if batch_i < n_batches - 1:
                scaler.scale(loss).backward(retain_graph=True)
            else:
                scaler.scale(loss).backward()

            del student_pred, student_z0_teacher_space


        # gc.collect()
        # torch.cuda.empty_cache()

        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)

        params = [p for group in optimizer.param_groups for p in group['params']]
        total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

        scaler.step(optimizer)

        scaler.update()

        loss_history.append(loss.item())

        wb.log({"n_inv_iters": args.n_inv_iters, "loss": loss, "lr": optimizer.param_groups[0]['lr'], "t": timestep.item(), "loss_w": w_t,
        "total_grad_norm": total_norm})

        scheduler.step()


        # Log and print progress
        if (distill_i + 1) % 1 == 0:
            print(
                "distill iteration: {}/{}, timestep: {}/{} total iter: {}/{} loss: {:.6e} lr: {:.6e}".format(
                    distill_i + 1,
                    n_distill_iters,
                    time_i + 1,
                    n_distill_timesteps,
                    iter_count + 1,
                    num_iters,
                    loss.item(),
                    optimizer.param_groups[0]['lr']
                )
            )

        #early_stopping = False
        early_stopping = check_for_early_stopping(loss_history, convergence_patience, convergence_threshold)
        if early_stopping:
            print("Early stopping at distill iteration: {}".format(distill_i + 1))
        
        is_last_iter_for_timestep = early_stopping or (distill_i == n_distill_iters-1)

        if (lr > 1e-14) and ((iter_count == 0) or is_last_iter_for_timestep):
            log_visualizations(student, teacher, latent, student_z0s_teacher_space, z_ts, pred_z0s, wb, timestep, teacher_timestep)

        iter_count += 1

        if is_last_iter_for_timestep:
            break
            

    # EulerEDM step to next timestep
    with torch.no_grad():
        if student_name == "blank_canvas":
            continue
        if time_i == n_timesteps - 1: # finished
            break 

        distilled_latent, _ = student.euler_edm_step(latent, student_t, next_t, cfg_w)
        #latent = undistilled_latent + args.distillation_guidance_scale * (distilled_latent - undistilled_latent)
        latent = distilled_latent

        latent_frames = student.decode(latent, type="np")
        print(f"t = {int(next_t[0])}")
        plot_frames(latent_frames, wb, "latent", "t={}".format(int(next_t[0])))



with torch.no_grad():
    del student, optimizer # free memory
    gc.collect()
    torch.cuda.empty_cache()

    x0 = teacher.decode(latent_translator(z0_student).to(dtype=teacher.dtype), type="pt", do_postprocess=False)
    x0 = F.interpolate(x0, size=(576,576), mode='bilinear', align_corners=False)

    x0_np = teacher.decode(latent_translator(z0_student).to(dtype=teacher.dtype), "np")
    plot_frames(x0_np, wb, "final_z0", "t=0", save_as_pdf=False, save_individual_pngs=False)

    # free memory, teacher still used later
    teacher.unet.cpu()
    teacher.vae.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    seva = SevaPipeline(device=device, value_dict=value_dict, do_compile=args.do_compile)
    seva.model.eval() 
    #seva.model.requires_grad_(True) # for guidance

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        z0 = seva.ae.encode(x0.cuda())

    timestep = t_min_distill * s_in
    eps = torch.randn_like(z0, device="cuda")
    sample_dt = args.sample_dt
    n_sample_timesteps = t_min_distill // sample_dt

    
    latent = seva.noise_to_timestep(z0, timestep, eps)

    latent_frames = seva.decode(latent, type="np")
    plot_frames(latent_frames, wb, "noisy_latent", "t={}".format(int(timestep[0])))

    #latent = seva.euler_edm_sample(latent, timestep, n_sample_timesteps, cfg_w)

# with grad due to guidance
latent = seva.euler_edm_sample_guided_latents(latent, timestep, n_sample_timesteps, cfg_w, reference_latent=z0, guidance_scale=args.sampling_guidance_scale)

with torch.no_grad():
    latent_frames = seva.decode(latent, type="np")
    latent_frames_out = seva.get_only_output_frames(latent_frames)
    plot_frames(latent_frames_out, wb, "final_res", "t=0", save_as_pdf=False)
    
    del seva # free memory
    gc.collect()
    torch.cuda.empty_cache()


    # final quality eval
    if teacher_name == "pix2pix":
        run_metric_suite_editing(np.asarray(pil_gt), np.asarray(latent_frames_out), text_target, edited_scene_prompt, wb)
    else:
        measurments = teacher.forward_operator(pil_gt) # measurements with original (seva) resolution
        run_metric_suite_controlled(np.asarray(latent_frames_out), np.asarray(measurments), teacher.forward_operator, wb)
        
    plot_frames(teacher.forward_operator(latent_frames_out), wb, "forward_op(final_res)", "", save_as_pdf=False)

    plot_logs(wb, output_log)
