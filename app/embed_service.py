from typing import List

import numpy as np

from app.settings import (
    has_cuda,
    embedding_model,
    model_folder,
    timing_decorator,
    has_mps,
)


class EmbeddingService:
    @timing_decorator
    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self._st = SentenceTransformer(
            embedding_model,
            device="cuda" if has_cuda else "mps" if has_mps else "cpu",
            cache_folder=model_folder,
        )

        if embedding_model.startswith("BAAI"):
            self._prefix_embed_query = (
                "Represent this sentence for searching relevant passages: "
            )
        elif embedding_model.startswith("intfloat"):
            self._prefix_embed_query = "query: "
        else:
            self._prefix_embed_query = ""

        if embedding_model.startswith("intfloat"):
            self._prefix_embed_document = "passage: "
        else:
            self._prefix_embed_document = ""

    def embed(self, sentences: List[str]):
        return self._st.encode(
            sentences, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float16)

    def embed_query(self, queries: List[str]):
        return self.embed([f"{self._prefix_embed_query}{q}" for q in queries])

    def embed_document(self, documents: List[str]):
        return self.embed([f"{self._prefix_embed_document}{q}" for q in documents])
