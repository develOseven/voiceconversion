import os
from typing import get_args

from voiceconversion.const import EmbedderType
from voiceconversion.embedder.Embedder import Embedder
from voiceconversion.embedder.OnnxEmbedder import OnnxEmbedder
from voiceconversion.downloader.WeightDownloader import (
    CONTENT_VEC_500_ONNX,
    SPIN_BASE,
    SPIN_V2,
)

import logging

logger = logging.getLogger(__name__)


class EmbedderManager:
    embedder: Embedder | None = None

    @classmethod
    def initialize(cls):
        pass

    @classmethod
    def get_embedder(
        cls, pretrain_dir: str, embedder_type: EmbedderType, force_reload: bool = False
    ) -> Embedder:
        if (
            cls.embedder is not None
            and cls.embedder.matchCondition(embedder_type)
            and not force_reload
        ):
            logger.info("Reusing embedder.")
            return cls.embedder
        cls.embedder = cls.load_embedder(pretrain_dir, embedder_type)
        return cls.embedder

    @classmethod
    def load_embedder(cls, pretrain_dir: str, embedder_type: EmbedderType) -> Embedder:
        logger.info(f"Loading embedder {embedder_type}")

        if embedder_type not in get_args(EmbedderType):
            raise RuntimeError(f"Unsupported embedder type: {embedder_type}")
        if embedder_type == "hubert_base" or embedder_type == "contentvec":
            file = CONTENT_VEC_500_ONNX
        elif embedder_type == "spin_base":
            file = SPIN_BASE
        elif embedder_type == "spin_v2":
            file = SPIN_V2

        return OnnxEmbedder().load_model(embedder_type, os.path.join(pretrain_dir, file))
