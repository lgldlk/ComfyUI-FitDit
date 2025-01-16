from typing import Tuple

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class PoseGuider(ModelMixin):
    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int] = (16, 32, 96, 256),
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(
            conditioning_channels, block_out_channels[0], kernel_size=3, padding=1
        )

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(
                nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1)
            )
            self.blocks.append(
                nn.Conv2d(
                    channel_in, channel_out, kernel_size=3, padding=1, stride=2
                )
            )
        
        self.out = zero_module(
            nn.Linear(
                block_out_channels[-1]*4,
                conditioning_embedding_channels,
            )
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)
        embedding = rearrange(embedding, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        embedding = self.out(embedding)

        return embedding

if __name__ == "__main__":
    import torch
    model = PoseGuider(conditioning_embedding_channels=3072, block_out_channels = (16, 32, 96, 256))
    inp = torch.randn((4, 3, 1024, 768))
    out = model(inp)

