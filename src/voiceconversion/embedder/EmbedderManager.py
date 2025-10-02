from voiceconversion.const import EmbedderType
from voiceconversion.embedder.Embedder import Embedder
from voiceconversion.embedder.OnnxContentvec import OnnxContentvec
import logging
logger = logging.getLogger(__name__)

class EmbedderManager:
    embedder: Embedder | None = None

    @classmethod
    def initialize(cls):
        pass

    @classmethod
    def get_embedder(cls, content_vec_500_onnx: str, embedder_type: EmbedderType, force_reload: bool = False) -> Embedder:
        if cls.embedder is not None \
            and cls.embedder.matchCondition(embedder_type) \
            and not force_reload:
            logger.info('Reusing embedder.')
            return cls.embedder
        cls.embedder = cls.load_embedder(content_vec_500_onnx, embedder_type)
        return cls.embedder

    @classmethod
    def load_embedder(cls, content_vec_500_onnx: str, embedder_type: EmbedderType) -> Embedder:
        logger.info(f'Loading embedder {embedder_type}')

        if embedder_type not in ["hubert_base", "contentvec"]:
            raise RuntimeError(f'Unsupported embedder type: {embedder_type}')
        file = content_vec_500_onnx
        return OnnxContentvec().load_model(file)
