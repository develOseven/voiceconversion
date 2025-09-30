import logging
from typing import Protocol
from const import PitchExtractorType
from voiceconversion.pitch_extractor.CrepeOnnxPitchExtractor import CrepeOnnxPitchExtractor
from voiceconversion.pitch_extractor.CrepePitchExtractor import CrepePitchExtractor
from voiceconversion.pitch_extractor.PitchExtractor import PitchExtractor
from voiceconversion.pitch_extractor.RMVPEOnnxPitchExtractor import RMVPEOnnxPitchExtractor
from voiceconversion.pitch_extractor.RMVPEPitchExtractor import RMVPEPitchExtractor
from voiceconversion.pitch_extractor.FcpePitchExtractor import FcpePitchExtractor
from voiceconversion.pitch_extractor.FcpeOnnxPitchExtractor import FcpeOnnxPitchExtractor


logger = logging.getLogger(__name__)


class PitchExtractorManager(Protocol):
    pitch_extractor: PitchExtractor | None = None

    @classmethod
    def getPitchExtractor(cls, pitch_extractor: PitchExtractorType, force_reload: bool) -> PitchExtractor:
        cls.pitch_extractor = cls.loadPitchExtractor(pitch_extractor, force_reload)
        return cls.pitch_extractor

    @classmethod
    def loadPitchExtractor(cls, pitch_extractor: PitchExtractorType, force_reload: bool) -> PitchExtractor:
        if cls.pitch_extractor is not None \
            and pitch_extractor == cls.pitch_extractor.type \
            and not force_reload:
            logger.info('Reusing pitch extractor.')
            return cls.pitch_extractor

        logger.info(f'Loading pitch extractor {pitch_extractor}')
        try:
            if pitch_extractor == 'crepe_tiny':
                return CrepePitchExtractor(pitch_extractor, 'pretrain/crepe_tiny.pth')
            elif pitch_extractor == 'crepe_full':
                return CrepePitchExtractor(pitch_extractor, 'pretrain/crepe_full.pth')
            elif pitch_extractor == "crepe_tiny_onnx":
                return CrepeOnnxPitchExtractor(pitch_extractor, 'pretrain/crepe_onnx_tiny.onnx')
            elif pitch_extractor == "crepe_full_onnx":
                return CrepeOnnxPitchExtractor(pitch_extractor, 'pretrain/crepe_onnx_full.onnx')
            elif pitch_extractor == "rmvpe":
                return RMVPEPitchExtractor('pretrain/rmvpe.pt')
            elif pitch_extractor == "rmvpe_onnx":
                return RMVPEOnnxPitchExtractor('pretrain/rmvpe.onnx')
            elif pitch_extractor == "fcpe":
                return FcpePitchExtractor('pretrain/fcpe.pt')
            elif pitch_extractor == "fcpe_onnx":
                return FcpeOnnxPitchExtractor('pretrain/fcpe.onnx')
            else:
                logger.warning(f"PitchExctractor not found {pitch_extractor}. Fallback to rmvpe_onnx")
                return RMVPEOnnxPitchExtractor('pretrain/rmvpe.onnx')
        except RuntimeError as e:
            logger.error(f'Failed to load {pitch_extractor}. Fallback to rmvpe_onnx.')
            logger.exception(e)
            return RMVPEOnnxPitchExtractor('pretrain/rmvpe.onnx')
