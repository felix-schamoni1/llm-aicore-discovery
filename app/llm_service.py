from typing import List, Dict
from app.settings import llm_model, has_cuda, timing_decorator, has_mps


class LLMService:
    @timing_decorator
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self._llm = AutoModelForCausalLM.from_pretrained(
            llm_model,
            device_map="auto" if has_cuda else "mps" if has_mps else "cpu",
            torch_dtype=torch.float16 if (has_cuda or has_mps) else torch.float32,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(llm_model)
        self._device = "cuda" if has_cuda else "mps" if has_mps else "cpu"

    def complete(self, data: List[Dict[str, str]], **kwargs) -> str:
        encoded_values = self._tokenizer.apply_chat_template(
            data, return_tensors="pt"
        ).to(self._device)

        tokens = encoded_values.shape[-1]

        generated_ids = self._llm.generate(encoded_values, **kwargs)
        return self._tokenizer.batch_decode(generated_ids[:, tokens:])[0]
