from diffusion_pipeline import DiffusionPipeline, SevaPipeline
from overrides import override
import torch
import itertools

# seva imports
from seva.seva.utils import load_model
from seva.seva.model import SGMWrapper
from seva.seva.modules.autoencoder import AutoEncoder
from seva.seva.modules.conditioner import CLIPConditioner
from seva.seva.sampling import DDPMDiscretization, DiscreteDenoiser, MultiviewCFG, append_dims
from seva.seva.eval import unload_model
from lora_diffusion import inject_trainable_lora, inject_trainable_lora_extended
from einops import repeat

from plot_utils import seva_tensor_to_np_plottable




class BlankCanvasStudent:
    def __init__(self, latent_shape, device):
        self.theta = torch.randn(latent_shape, requires_grad=True, device=device, dtype=torch.float16)
        self.latent_shape = latent_shape
        self.discretization = DDPMDiscretization() # just for compatability with mv distillation script, has no meaning

    def get_trainable_parameters(self):
        return [self.theta]

    def predict_sample(self):
        return self.theta
    
    def get_latent_shape(self):
        return self.latent_shape

    def predict_eps_and_sample(self, z_t, timestep, guidance_scale):
        return None, self.predict_sample()


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



class SevaStudent(SevaPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @override
    def set_model_requires_grad(self):
        self.model.requires_grad_(True)

    def get_trainable_parameters(self):
        return self.model.parameters()


class SevaFrozenMVTransformerStudent(SevaPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @override
    def set_model_requires_grad(self):
        self.model.requires_grad_(True)

        # Ensure MultiViewTransformer params remain frozen
        for block in self.model.module.input_blocks:
            for layer in block:
                if layer.__class__.__name__ == "MultiviewTransformer":
                    #print(f"Freezing kayer: {layer}")
                    for param in layer.parameters():
                        param.requires_grad = False
    
        for layer in self.model.module.middle_block:
            if layer.__class__.__name__ == "MultiviewTransformer":
                #print(f"Freezing layer: {layer}")
                for param in layer.parameters():
                    param.requires_grad = False

        for block in self.model.module.output_blocks:
            for layer in block:
                if layer.__class__.__name__ == "MultiviewTransformer":
                    #print(f"Freezing layer: {layer}")
                    for param in layer.parameters():
                        param.requires_grad = False

                    
    def get_trainable_parameters(self):
        return (p for n, p in self.model.named_parameters() if p.requires_grad)


class SevaLoRAStudent(SevaStudent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @override
    def set_model_requires_grad(self):
         # Turning off trainable parameters, except for the first layer (due to LoRA bug)
        self.model.requires_grad_(False)
        first_layer = self.model.module.input_blocks[0]
        for _, param in first_layer.named_parameters():
            param.requires_grad = True
        
        self.model_lora_params, names = inject_trainable_lora_extended(self.model, target_replace_module=["TimestepEmbedSequential", "Sequential", "ResBlock", "Upsample", "MultiviewTransformer", "Downsample"])
        print(f"Lora layers: {names}")

    @override
    def get_trainable_parameters(self):
        return itertools.chain(*self.model_lora_params) 

   

def get_mv_student(student_name, **kwargs):
    if student_name == "blank_canvas":
        return BlankCanvasStudent(device=kwargs["device"], latent_shape=kwargs["latent_shape"])
    if student_name == "seva":
        return SevaStudent(device=kwargs["device"], value_dict=kwargs["value_dict"], do_compile=kwargs["do_compile"])
    elif student_name == "seva_frozen_mv_transformer":
        return SevaFrozenMVTransformerStudent(device=kwargs["device"], value_dict=kwargs["value_dict"], do_compile=kwargs["do_compile"])
    elif student_name == "seva_lora":
        return SevaLoRAStudent(device=kwargs["device"], value_dict=kwargs["value_dict"], do_compile=kwargs["do_compile"])  
    else:
        raise ValueError(f"Unknown multiview student name: {student_name}")

