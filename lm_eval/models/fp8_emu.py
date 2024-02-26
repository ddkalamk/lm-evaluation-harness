import torch

from lm_eval.api.registry import register_model
from lm_eval.models.utils import get_dtype

from .huggingface import HFLM


@register_model("fp8_emu")
class Fp8EmuLM(HFLM):
    def __init__(self, *args, **kwargs):
        torch.hfloat8 = torch.float8_e4m3fn
        torch.bfloat8 = torch.float8_e5m2
        weight_dtype = kwargs.pop("weight_dtype", None)
        # print(f"Using FP8 emulation - weight_dtype: {weight_dtype}")
        super().__init__(*args, **kwargs)
        if weight_dtype is not None:
            weight_dtype = get_dtype(weight_dtype)
            for n, m in self.model.named_modules():
                if isinstance(m, torch.nn.Linear) and "lm_head" not in n:
                    print(f"Emulating {weight_dtype} weights for Module {n}")
                    orig_dtype = m.weight.dtype
                    m.weight.data = m.weight.data.to(weight_dtype).to(orig_dtype)
