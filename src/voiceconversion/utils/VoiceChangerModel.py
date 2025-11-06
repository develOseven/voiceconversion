from typing import Any, Protocol, TypeAlias

import numpy as np
import torch

from voiceconversion.const import VoiceChangerType
from voiceconversion.voice_changer_settings import VoiceChangerSettings

AudioInOutFloat: TypeAlias = np.ndarray[Any, np.dtype[np.float32]]


class VoiceChangerModel(Protocol):
    voiceChangerType: VoiceChangerType

    def __init__(self, settings: VoiceChangerSettings): ...

    def initialize(self, force_reload: bool, pretrain_dir: str): ...

    def convert(self, data: torch.Tensor, sample_rate: int) -> torch.Tensor: ...

    def inference(self, data: AudioInOutFloat) -> torch.Tensor: ...

    def realloc(
        self,
        block_frame: int,
        extra_frame: int,
        crossfade_frame: int,
        sola_search_frame: int,
    ): ...

    def export2onnx() -> Any: ...
