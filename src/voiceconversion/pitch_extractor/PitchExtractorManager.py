import os
import logging
from typing import Protocol
from voiceconversion.const import PitchExtractorType
from voiceconversion.pitch_extractor.CrepeOnnxPitchExtractor import CrepeOnnxPitchExtractor
from voiceconversion.pitch_extractor.CrepePitchExtractor import CrepePitchExtractor
from voiceconversion.pitch_extractor.PitchExtractor import PitchExtractor
from voiceconversion.pitch_extractor.RMVPEOnnxPitchExtractor import RMVPEOnnxPitchExtractor
from voiceconversion.pitch_extractor.RMVPEPitchExtractor import RMVPEPitchExtractor
from voiceconversion.pitch_extractor.FcpePitchExtractor import FcpePitchExtractor
from voiceconversion.pitch_extractor.FcpeOnnxPitchExtractor import FcpeOnnxPitchExtractor
from voiceconversion.downloader.WeightDownloader import (
    CREPE_ONNX_FULL,
    CREPE_ONNX_TINY,
    CREPE_FULL,
    CREPE_TINY,
    RMVPE,
    RMVPE_ONNX,
    FCPE,
    FCPE_ONNX,
)


logger = logging.getLogger(__name__)


class PitchExtractorManager(Protocol):
    pitch_extractor: PitchExtractor | None = None

    @classmethod
    def getPitchExtractor(cls, pitch_extractor: PitchExtractorType, force_reload: bool, pretrain_dir: str) -> PitchExtractor:
        cls.pitch_extractor = cls.loadPitchExtractor(pitch_extractor, force_reload, pretrain_dir)
        return cls.pitch_extractor

    @classmethod
    def loadPitchExtractor(cls, pitch_extractor: PitchExtractorType, force_reload: bool, pretrain_dir: str) -> PitchExtractor:
        if cls.pitch_extractor is not None \
            and pitch_extractor == cls.pitch_extractor.type \
            and not force_reload:
            logger.info('Reusing pitch extractor.')
            return cls.pitch_extractor

        logger.info(f'Loading pitch extractor {pitch_extractor}')
        try:
            if pitch_extractor == 'crepe_tiny':
                return CrepePitchExtractor(pitch_extractor, os.path.join(pretrain_dir, CREPE_TINY))
            elif pitch_extractor == 'crepe_full':
                return CrepePitchExtractor(pitch_extractor, os.path.join(pretrain_dir, CREPE_FULL))
            elif pitch_extractor == "crepe_tiny_onnx":
                return CrepeOnnxPitchExtractor(pitch_extractor, os.path.join(pretrain_dir, CREPE_ONNX_TINY))
            elif pitch_extractor == "crepe_full_onnx":
                return CrepeOnnxPitchExtractor(pitch_extractor, os.path.join(pretrain_dir, CREPE_ONNX_FULL))
            elif pitch_extractor == "rmvpe":
                return RMVPEPitchExtractor(os.path.join(pretrain_dir, RMVPE))
            elif pitch_extractor == "rmvpe_onnx":
                return RMVPEOnnxPitchExtractor(os.path.join(pretrain_dir, RMVPE_ONNX))
            elif pitch_extractor == "fcpe":
                return FcpePitchExtractor(os.path.join(pretrain_dir, FCPE))
            elif pitch_extractor == "fcpe_onnx":
                return FcpeOnnxPitchExtractor(os.path.join(pretrain_dir, FCPE_ONNX))
            else:
                logger.warning(f"PitchExctractor not found {pitch_extractor}. Fallback to rmvpe_onnx")
                return RMVPEOnnxPitchExtractor(os.path.join(pretrain_dir, RMVPE_ONNX))
        except RuntimeError as e:
            logger.error(f'Failed to load {pitch_extractor}. Fallback to rmvpe_onnx.')
            logger.exception(e)
            return RMVPEOnnxPitchExtractor(os.path.join(pretrain_dir, RMVPE_ONNX))
