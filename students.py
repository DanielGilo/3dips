from diffusion_pipeline import DiffusionPipeline
from overrides import override
import torch


class BlankCanvasStudent:
    def __init__(self, latent_shape, device):
        self.theta = torch.randn(latent_shape, requires_grad=True, device=device).float()

    def get_trainable_params(self):
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

