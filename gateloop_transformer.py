import torch
from torch import nn
from torch.nn import Module, ModuleList

class RMSNorm(Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class GateLoopAttention(Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
    
class FeedForward(Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x

class Transformer(Module):
    def __init__(
        self,
        dim,
        *,
        num_tokens,
        depth,
    ):
        super().__init__()
        self.to_tokens = nn.Embedding(num_tokens, dim)

        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(
                ModuleList([
                    RMSNorm(),
                    GateLoopAttention(),
                    RMSNorm(),
                    FeedForward()
                ])
            )

        self.to_logits = nn.Sequential(
            RMSNorm(),
            nn.Linear(dim, num_tokens, bias = False)
        )

    def forward(self, x):
        x = self.to_tokens(x)
        
        for attn_norm, attn, ff_norm, ff in self.layers:
            attn_input = attn_norm(x)
            x = attn(attn_input) + x

            ff_input = ff_norm(x)
            x = ff(ff_input) + x

        return self.to_logits(x)