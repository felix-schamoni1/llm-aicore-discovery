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

embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
llm_model = os.getenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.1")

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
        logging.info(f"%s took %.6f seconds to execute.", func, elapsed_time)
        return result

    return wrapper
