from dataclasses import dataclass
from typing import Literal, TypeAlias

from voiceconversion.const import VoiceChangerType

ImportModelParamFileKind: TypeAlias = Literal[
    "rvcModel",
    "rvcIndex",
]


@dataclass
class ImportModelParamFile:
    name: str
    kind: ImportModelParamFileKind
    dir: str


@dataclass
class ImportModelParams:
    voice_changer_type: VoiceChangerType
    files: list[ImportModelParamFile]
    params: dict
