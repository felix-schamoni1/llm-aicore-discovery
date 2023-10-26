import functools
import logging
import os
import sys
import time
from pathlib import Path

import torch

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s - %(name)-s %(levelname)-s - %(message)s",
)
has_cuda = torch.cuda.is_available()
has_mps = torch.backends.mps.is_available()

embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
llm_model = os.getenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.1")

true_vals = ["1", "y", "Y"]
is_ai_core = os.getenv("AI_CORE", "0") in true_vals

http_prefix = "" if not is_ai_core else "/v1"
use_flash_attn = os.getenv("USE_FLASH_ATTN", "0") in true_vals


if has_cuda:
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_properties(i))

root_folder = Path(__file__).parent.parent.resolve()
model_folder = root_folder / "models"
model_folder.mkdir(exist_ok=True)

os.environ["TRANSFORMERS_CACHE"] = str(model_folder)


def timing_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        logging.info(f"%s took %.4f seconds to execute.", func, elapsed_time)
        return result

    return wrapper


def disable_grad():
    import torch

    torch.set_grad_enabled(False)
