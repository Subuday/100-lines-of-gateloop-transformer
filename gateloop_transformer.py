import torch
from torch import nn
from torch.nn import Module, ModuleList
from einops import einsum, rearrange
from associative_scan import associative_scan

class RMSNorm(Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class GateLoopAttention(Module):

    def __init__(
        self,
        dim: int,
        *,
        dim_inner: int,
        num_heads: int,
    ):
        super().__init__()
        self.dim_inner = dim_inner
        self.num_heads = num_heads
        assert (dim_inner % num_heads) == 0, f'dimension {dim_inner} must be divisible by number of gate loop heads {num_heads}'

        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)
        self.to_a = nn.Linear(dim, num_heads * 2)

        self.out_proj = nn.Linear(dim_inner, dim, bias = False) if dim_inner != dim else nn.Identity()

    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = self.num_heads), (q, k, v))

        a = self.to_a(x)
        a = rearrange(a, 'b n (h c) -> (b h) n 1 1 c', h = self.num_heads, c = 2)
        a = torch.view_as_complex(a)

        magnitude, phase = a.abs(), a.angle()
        a = torch.polar(magnitude.sigmoid(), phase)

        out = self._gate_loop(q, k, v, a)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = self.num_heads)
        return self.out_proj(out)

    @staticmethod
    def _gate_loop(q, k, v, a):
        kv = einsum(k, v, 'b n d, b n e -> b n d e')
        kv = kv + 0.j

        def binary_operator(a, b):
            a_i, kv_i = a
            a_j, kv_j = b
            return a_j * a_i, a_j * kv_i + kv_j

        _, kv = associative_scan(binary_operator, (a, kv))

        return einsum('b n d, b n d e -> b n e', q, kv.real)
    
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
                    GateLoopAttention(
                        dim,
                        dim_inner = 256,
                        num_heads = 4
                    ),
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