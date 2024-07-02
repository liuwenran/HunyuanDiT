
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin
from .embedders import PatchEmbed

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class PoseGuider(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        input_size=(32, 32),
        patch_size=2,
        in_channels=4,
        hidden_size=1408,
    ):
        super().__init__()
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        self.before_proj = zero_module(nn.Linear(hidden_size, hidden_size))


    def forward(self, condition):
        condition = self.x_embedder(condition)
        embedding = self.before_proj(condition) # add condition

        return embedding
