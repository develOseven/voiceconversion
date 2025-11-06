# from const import PitchExtractorType
import logging
from dataclasses import dataclass, field
from typing import NamedTuple

from voiceconversion.const import VoiceChangerType
from voiceconversion.data.imported_model_info import RVCImportedModelInfo

logger = logging.getLogger(__name__)


class SetPropertyResult(NamedTuple):
    error: bool
    old_value: str | int | float | bool | None


@dataclass
class VoiceChangerSettings:
    # General settings

    outputSampleRate: int = 48000
    inputSampleRate: int = 48000

    crossFadeOverlapSize: float = 0.1
    serverReadChunkSize: int = 192
    extraConvertSize: float = 0.5
    gpu: int = -1
    forceFp32: int = 0
    disableJit: int = 0

    # RVCv2 settings
    dstId: int = 0

    f0Detector: str = "rmvpe_onnx"

    silentThreshold: int = -90

    silenceFront: int = 1

    rvcImportedModelInfo: RVCImportedModelInfo = field(
        default_factory=lambda: RVCImportedModelInfo()
    )


# class VoiceChangerSettings:

#     def __eq__(self, other):
#         return self.get_properties() == other.get_properties()

#     def to_dict(self) -> dict:
#         return self.get_properties()

#     def get_properties(self) -> dict:
#         return {
#             key: value.fget(self)
#             for key, value in self.__class__.__dict__.items()
#             if isinstance(value, property)
#         }

#     def set_properties(self, data: dict) -> list[SetPropertyResult]:
#         return [self.set_property(key, value) for key, value in data.items()]

#     def set_property(self, key, value) -> SetPropertyResult:
#         cls = self.__class__
#         if key in IGNORED_KEYS:
#             return SetPropertyResult(error=False, old_value=None)
#         if key not in cls.__dict__:
#             logger.error(f"Failed to set setting: {key} does not exist")
#             return SetPropertyResult(error=True, old_value=None)
#         p = cls.__dict__[key]
#         if not isinstance(p, property):
#             return SetPropertyResult(error=True, old_value=None)
#         if p.fset is None:
#             logger.error(f"Failed to set setting: {key} is immutable.")
#             return SetPropertyResult(error=True, old_value=None)
#         old_value = p.fget(self)
#         p.fset(self, value)
#         return SetPropertyResult(error=False, old_value=old_value)

#     def get_property(self, key):
#         return getattr(self, key)

#     # Immutable
#     _version: str = "v1"

#     @property
#     def version(self):
#         return self._version

#     # General settings

#     _dir: str = ""
#     _voiceChangerType: VoiceChangerType = "None"

#     _outputSampleRate: int = 48000
#     _inputSampleRate: int = 48000

#     _crossFadeOverlapSize: float = 0.1
#     _serverReadChunkSize: int = 192
#     _extraConvertSize: float = 0.5
#     _gpu: int = -1
#     _forceFp32: int = 0
#     _disableJit: int = 0

#     @property
#     def dir(self):
#         return self._dir

#     @dir.setter
#     def dir(self, dir: str):
#         self._dir = dir

#     @property
#     def voiceChangerType(self):
#         return self._voiceChangerType

#     @voiceChangerType.setter
#     def voiceChangerType(self, voice_changer_type: VoiceChangerType):
#         self._voiceChangerType = voice_changer_type

#     @property
#     def inputSampleRate(self):
#         return self._inputSampleRate

#     @inputSampleRate.setter
#     def inputSampleRate(self, sample_rate: str):
#         self._inputSampleRate = int(sample_rate)

#     @property
#     def outputSampleRate(self):
#         return self._outputSampleRate

#     @outputSampleRate.setter
#     def outputSampleRate(self, sample_rate: str):
#         self._outputSampleRate = int(sample_rate)

#     @property
#     def gpu(self):
#         return self._gpu

#     @gpu.setter
#     def gpu(self, gpu: str):
#         self._gpu = int(gpu)

#     @property
#     def extraConvertSize(self):
#         return self._extraConvertSize

#     @extraConvertSize.setter
#     def extraConvertSize(self, size: str):
#         self._extraConvertSize = float(size)

#     @property
#     def serverReadChunkSize(self):
#         return self._serverReadChunkSize

#     @serverReadChunkSize.setter
#     def serverReadChunkSize(self, size: str):
#         self._serverReadChunkSize = int(size)

#     @property
#     def crossFadeOverlapSize(self):
#         return self._crossFadeOverlapSize

#     @crossFadeOverlapSize.setter
#     def crossFadeOverlapSize(self, size: str):
#         self._crossFadeOverlapSize = float(size)

#     @property
#     def forceFp32(self):
#         return self._forceFp32

#     @forceFp32.setter
#     def forceFp32(self, enable: str):
#         self._forceFp32 = int(enable)

#     @property
#     def disableJit(self):
#         return self._disableJit

#     @disableJit.setter
#     def disableJit(self, enable: str):
#         self._disableJit = int(enable)

#     # RVCv2 settings
#     _dstId: int = 0

#     _f0Detector: str = "rmvpe_onnx"
#     _tran: int = 0
#     _formantShift: float = 0
#     _useONNX: int = 0
#     _modelFileOnnx: str = ""

#     _silentThreshold: int = -90

#     _indexRatio: float = 0
#     _protect: float = 0.5
#     _silenceFront: int = 1

#     @property
#     def dstId(self):
#         return self._dstId

#     @dstId.setter
#     def dstId(self, id: str):
#         self._dstId = int(id)

#     @property
#     def f0Detector(self):
#         return self._f0Detector

#     @f0Detector.setter
#     def f0Detector(self, pitch_extractor_type: str):
#         self._f0Detector = pitch_extractor_type

#     @property
#     def tran(self):
#         return self._tran

#     @tran.setter
#     def tran(self, tone: str):
#         self._tran = int(tone)

#     @property
#     def formantShift(self):
#         return self._formantShift

#     @formantShift.setter
#     def formantShift(self, shift_size: str):
#         self._formantShift = float(shift_size)

#     @property
#     def useONNX(self):
#         return self._useONNX

#     @useONNX.setter
#     def useONNX(self, enabled: str):
#         self._useONNX = int(enabled)

#     @property
#     def modelFileOnnx(self):
#         return self._modelFileOnnx

#     @modelFileOnnx.setter
#     def modelFileOnnx(self, modelFileOnnx: str):
#         self._modelFileOnnx = modelFileOnnx

#     @property
#     def silentThreshold(self):
#         return self._silentThreshold

#     @silentThreshold.setter
#     def silentThreshold(self, threshold: str):
#         self._silentThreshold = int(threshold)

#     @property
#     def indexRatio(self):
#         return self._indexRatio

#     @indexRatio.setter
#     def indexRatio(self, ratio: str):
#         self._indexRatio = float(ratio)

#     @property
#     def protect(self):
#         return self._protect

#     @protect.setter
#     def protect(self, protect: str):
#         self._protect = float(protect)

#     @property
#     def silenceFront(self):
#         return self._silenceFront

#     @silenceFront.setter
#     def silenceFront(self, enable: str):
#         self._silenceFront = int(enable)
