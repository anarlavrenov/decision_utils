import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(
            self,
            base: nn.Linear,
            rank: int,
            alpha: float,
            dropout: float = 0.0
    ):
        super().__init__()

        self.base = base
        self.rank = rank
        self.alpha = alpha

        in_ = self.base.in_features
        out_ = self.base.out_features
        dev, dt = self.base.weight.device, self.base.weight.dtype

        self.lora_a = nn.Linear(in_, rank, bias=False, device=dev, dtype=dt)
        self.lora_b = nn.Linear(rank, out_, bias=False, device=dev, dtype=dt)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x):
        # B, L, out_.
        out = self.base(x)
        # B, L, rank.
        lora_out = self.lora_a(self.dropout(x))
        # B, L, out_.
        lora_out = self.lora_b(lora_out) * (self.alpha / self.rank)

        # B, L, out_.
        return out + lora_out


def apply_lora(model, rank, alpha, dropout=0.0):
    EXCLUDE = ("lm_head", "in_proj_a", "in_proj_b")
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and name.split(".")[-1] not in EXCLUDE:
            parent_name, child_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
            setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha))