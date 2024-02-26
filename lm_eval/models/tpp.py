import torch

from lm_eval.api.registry import register_model
from lm_eval.models.utils import get_dtype

from .huggingface import HFLM


@register_model("tpp")
class TppLM(HFLM):
    def __init__(self, *args, **kwargs):
        use_tpp = kwargs.pop("use_tpp", True)
        dtype = kwargs.get("dtype", "auto")
        weight_dtype = kwargs.pop("weight_dtype", None)
        cpp_profile = kwargs.pop("cpp_profile", False)
        use_jit = kwargs.pop("use_jit", True)
        only_last_logit = kwargs.pop("only_last_logit", False)
        num_beams = kwargs.get("num_beams", 1)
        tpp_dtype = get_dtype(dtype)
        print(f"Use TPP: {use_tpp}, weight_dtype: {weight_dtype}")
        super().__init__(*args, **kwargs)
        if use_tpp:
            from tpp_pytorch_extension.llm.llm_common import (
                jit_trace_model,
                optimize_for_first_token,
            )

            if weight_dtype is not None:
                weight_dtype = get_dtype(weight_dtype)
            device = torch.device("cpu")
            if self.model.config.architectures[0] == "GPTJForCausalLM":
                from tpp_pytorch_extension.llm.fused_gptj_infer import (
                    OptimizeModelForGPTJ,
                )

                OptimizeModelForGPTJ(
                    self.model,
                    dtype=tpp_dtype,
                    device=device,
                    weight_dtype=weight_dtype,
                )
            elif self.model.config.architectures[0] == "OPTForCausalLM":
                from tpp_pytorch_extension.llm.fused_opt_infer import (
                    OptimizeModelForOPT,
                )

                OptimizeModelForOPT(
                    self.model,
                    dtype=tpp_dtype,
                    device=device,
                    weight_dtype=weight_dtype,
                )
            elif self.model.config.architectures[0] == "LLaMAForCausalLM":
                from tpp_pytorch_extension.llm.fused_llama_infer import (
                    OptimizeModelForLlama,
                )

                OptimizeModelForLlama(
                    self.model,
                    dtype=tpp_dtype,
                    device=device,
                    weight_dtype=weight_dtype,
                )
            elif self.model.config.architectures[0] == "LlamaForCausalLM":
                from tpp_pytorch_extension.llm.fused_llama_infer import (
                    OptimizeModelForLlama,
                )

                OptimizeModelForLlama(
                    self.model,
                    dtype=tpp_dtype,
                    device=device,
                    weight_dtype=weight_dtype,
                )
            else:
                print(type(self.model.config.architectures))
                print(self.model.config.architectures)
                raise NotImplementedError("Model type not supported by TPP")

            if use_jit is True:
                self._model = jit_trace_model(
                    self.model,
                    self.tokenizer,
                    num_beams,
                    indirect_kv=True,
                    enable_profile=cpp_profile,
                    only_last_logit=only_last_logit,
                )
            else:
                self._model = optimize_for_first_token(
                    self.model,
                    num_beams,
                    enable_profile=cpp_profile,
                    only_last_logit=only_last_logit,
                )
