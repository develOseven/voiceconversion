from typing import Literal, TypeAlias


F0_MIN = 50
F0_MAX = 1100

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
