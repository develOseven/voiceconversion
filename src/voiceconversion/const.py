import numpy as np
import os
import sys
import tempfile
from enum import Enum
from typing import Literal, TypeAlias


VoiceChangerType: TypeAlias = Literal[
    "RVC",
    "None",
]

SERVER_DEVICE_SAMPLE_RATES = [16000, 32000, 44100, 48000, 96000, 192000]

EmbedderType: TypeAlias = Literal["hubert_base", "contentvec", "spin_base", "spin_v2"]


class EnumInferenceTypes(Enum):
    pyTorchRVC = "pyTorchRVC"
    pyTorchRVCNono = "pyTorchRVCNono"
    pyTorchRVCv2 = "pyTorchRVCv2"
    pyTorchRVCv2Nono = "pyTorchRVCv2Nono"
    pyTorchWebUI = "pyTorchWebUI"
    pyTorchWebUINono = "pyTorchWebUINono"
    onnxRVC = "onnxRVC"
    onnxRVCNono = "onnxRVCNono"


F0_MIN = 50
F0_MAX = 1100
F0_MEL_MIN = 1127 * np.log(1 + F0_MIN / 700)
F0_MEL_MAX = 1127 * np.log(1 + F0_MAX / 700)

PitchExtractorType: TypeAlias = Literal[
    "crepe_full",
    "crepe_tiny",
    "crepe_full_onnx",
    "crepe_tiny_onnx",
    "rmvpe",
    "rmvpe_onnx",
    "fcpe",
    "fcpe_onnx",
]

tmpdir = tempfile.TemporaryDirectory()
TMP_DIR = os.path.join(tmpdir.name, "tmp_dir") if hasattr(sys, "_MEIPASS") else "tmp_dir"

MAX_SLOT_NUM = 500
