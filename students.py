from diffusion_pipeline import DiffusionPipeline
from overrides import override
import torch

# seva imports
from seva.seva.utils import load_model
from seva.seva.model import SGMWrapper
from seva.seva.modules.autoencoder import AutoEncoder
from seva.seva.modules.conditioner import CLIPConditioner
from seva.seva.sampling import DDPMDiscretization, DiscreteDenoiser, MultiviewCFG, append_dims
from seva.seva.eval import unload_model
from lora_diffusion import inject_trainable_lora
from einops import repeat





class BlankCanvasStudent:
    def __init__(self, latent_shape, device):
        self.theta = torch.randn(latent_shape, requires_grad=True, device=device).float()

    def get_trainable_parameters(self):
        return [self.theta]

    def predict_sample(self):
        return self.theta


class DiffusionStudent(DiffusionPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_trainable_parameters(self):
        raise NotImplementedError


class SDStudent(DiffusionStudent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for p in self.unet.parameters():
            p.requires_grad = True
        self.unet.train()

    @override
    def get_trainable_parameters(self):
        return self.unet.parameters()


class SDLoRAStudent(DiffusionStudent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from peft import LoraConfig  # this import could be harmful
        unet_lora_config = LoraConfig(
            r=8,
            lora_alpha=8,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        self.unet.add_adapter(unet_lora_config)
        self.lora_layers = filter(lambda p: p.requires_grad, self.unet.parameters())
        self.unet.train()

    @override
    def get_trainable_parameters(self):
        return self.lora_layers





class SevaLoRAStudent:
    def __init__(self, device, value_dict):
        self.device = device
        self.model = SGMWrapper(load_model(device="cpu", verbose=True)).to(self.device)
        # loading to cpu to save memory
        self.ae = AutoEncoder(chunk_size=1).to("cpu")
        self.conditioner = CLIPConditioner().to("cpu")
        self.discretization = DDPMDiscretization()
        self.denoiser = DiscreteDenoiser(discretization=self.discretization, num_idx=1000, device=device)
        self.guider = MultiviewCFG(cfg_min=1.2) # from seva demo.py
        self.version_dict = {
            "H": 576,
            "W": 576,
            "T": 21,
            "C": 4,
            "f": 8,
            "options": {},
        }

        self.model.train()

        # Turning off trainable parameters, except for the first layer (due to LoRA bug)
        self.model.requires_grad_(False)
        first_layer = self.model.module.input_blocks[0]
        for name, param in first_layer.named_parameters():
            param.requires_grad = True
        
        self.model_lora_params, _ = inject_trainable_lora(self.model)


        self.value_dict = value_dict

        self.c, self.uc, self.additional_model_inputs, self.additional_sampler_inputs = self.prepare_model_inputs()

    def prepare_model_inputs(self):
        """
        adapted from seva's eval.py do_sample function.
        """
        
        imgs = self.value_dict["cond_frames"].to(self.device)
        input_masks = self.value_dict["cond_frames_mask"].to(self.device)
        pluckers = self.value_dict["plucker_coordinate"].to(self.device)

        T = self.version_dict["T"]

        num_samples = [1, T]
        encoding_t = 1 # not sure what that is
        with torch.inference_mode():
        #with torch.autocast("cuda"): # Elias
            self.ae = self.ae.to(self.device)
            self.conditioner = self.conditioner.to(self.device)

            input_latents = self.ae.encode(imgs[input_masks], encoding_t)
            latents = torch.nn.functional.pad(
                input_latents, (0, 0, 0, 0, 0, 1), value=1.0
            )
            c_crossattn = repeat(self.conditioner(imgs[input_masks]).mean(0), "d -> n 1 d", n=T)
            uc_crossattn = torch.zeros_like(c_crossattn)
            c_replace = latents.new_zeros(T, *latents.shape[1:])
            c_replace[input_masks] = latents
            uc_replace = torch.zeros_like(c_replace)
            c_concat= torch.cat(
                [
                    repeat(
                        input_masks,
                        "n -> n 1 h w",
                        h=pluckers.shape[2],
                        w=pluckers.shape[3],
                    ),
                    pluckers,
                ],
                1,
            )
            uc_concat = torch.cat(
                [pluckers.new_zeros(T, 1, *pluckers.shape[-2:]), pluckers], 1
            )
            c_dense_vector = pluckers
            uc_dense_vector = c_dense_vector
            c = {
                "crossattn": c_crossattn,
                "replace": c_replace,
                "concat": c_concat,
                "dense_vector": c_dense_vector,
            }
            uc = {
                "crossattn": uc_crossattn,
                "replace": uc_replace,
                "concat": uc_concat,
                "dense_vector": uc_dense_vector,
            }
            unload_model(self.ae)
            unload_model(self.conditioner)

            additional_model_inputs = {"num_frames": T}
            additional_sampler_inputs = {
                "c2w": self.value_dict["c2w"].to("cuda"),
                "K": self.value_dict["K"].to("cuda"),
                "input_frame_mask": self.value_dict["cond_frames_mask"].to("cuda")
            }

            return c, uc, additional_model_inputs, additional_sampler_inputs


    def get_trainable_parameters(self):
        return self.model_lora_params

    def noise_to_timestep(self, z0, timestep, eps):
        raise NotImplementedError # I think this function is not needed for a student

    def prepare_input_for_denoiser(self, z_t, sigma):
        input, sigma, c = self.guider.prepare_inputs(z_t, sigma, self.c, self.uc)
        return input, sigma, c
    
    def get_sigma_from_timestep(self, timestep):
        return self.denoiser.idx_to_sigma(timestep.to(dtype=torch.int))

    def predict_eps_and_sample(self, z_t, timestep, guidance_scale):        
        sigma_orig = self.get_sigma_from_timestep(timestep)
        input, sigma, c = self.prepare_input_for_denoiser(z_t, sigma_orig)
        
        pred_z0_c_and_uc = self.denoiser(self.model, input, sigma, c.copy(), **self.additional_model_inputs)

        # logic taken from seva's DiscreteDenoiser __call__() method, 
        # here we isolate the network output which is the predicted noise from the denoiser's return value.
        sigma = append_dims(sigma, input.ndim)
        if "replace" in c: # for input frames (mask = True) use the clean latents
            x, mask = c.pop("replace").split((input.shape[1], 1), dim=1)
            input = input * (1 - mask) + x * mask
        c_skip, c_out, _, _ = self.denoiser.scaling(sigma)
        pred_eps_c_and_uc = (input * c_skip - pred_z0_c_and_uc) / c_out

        pred_z0 = self.guider(pred_z0_c_and_uc, sigma_orig, guidance_scale, **self.additional_sampler_inputs)
        pred_eps = self.guider(pred_eps_c_and_uc, sigma_orig, guidance_scale, **self.additional_sampler_inputs)

       
        return pred_eps, pred_z0


    @torch.no_grad()
    def decode(self, latent):
        self.ae.to(self.device)
        samples = self.ae.decode(latent, 1)
        samples = (((samples.permute((0,2,3,1)) + 1) / 2.0)* 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        unload_model(self.ae)

        return samples
        