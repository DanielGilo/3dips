from diffusion_pipeline import DiffusionPipeline

class Teacher(DiffusionPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
