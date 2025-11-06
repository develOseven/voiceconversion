import json
import logging
import os
from dataclasses import asdict, dataclass, field

from voiceconversion.const import EmbedderType, EnumInferenceTypes, VoiceChangerType

logger = logging.getLogger(__name__)


@dataclass
class ImportedModelInfo:
    id: int = -1
    storageDir: str = ""
    voiceChangerType: VoiceChangerType | None = None
    name: str = ""
    description: str = ""
    credit: str = ""
    termsOfUseUrl: str = ""
    speakers: dict = field(default_factory=lambda: {})


@dataclass
class RVCImportedModelInfo(ImportedModelInfo):
    voiceChangerType: VoiceChangerType = "RVC"
    modelFile: str = ""
    modelFileOnnx: str = ""
    indexFile: str = ""
    defaultTune: int = 0
    defaultFormantShift: float = 0
    defaultIndexRatio: float = 0
    defaultProtect: float = 0.5
    isONNX: bool = False
    modelType: str = EnumInferenceTypes.pyTorchRVC.value
    modelTypeOnnx: str = EnumInferenceTypes.onnxRVC.value
    samplingRate: int = -1
    f0: bool = True
    embChannels: int = 256
    embOutputLayer: int = 9
    useFinalProj: bool = True
    deprecated: bool = False
    embedder: EmbedderType = "hubert_base"

    sampleId: str = ""
    speakers: dict = field(default_factory=lambda: {"0": "target"})

    version: str = "v2"


def load_imported_model_info(id: int, storage_dir: str) -> ImportedModelInfo | None:
    json_file = os.path.join(storage_dir, "params.json")
    if not os.path.exists(json_file):
        return None
    with open(json_file, encoding="utf-8") as f:
        json_dict = json.load(f)
    imported_model_info_key = list(ImportedModelInfo.__annotations__.keys())
    imported_model_info = ImportedModelInfo(
        id=id,
        storageDir=storage_dir,
        **{k: v for k, v in json_dict.items() if k in imported_model_info_key},
    )
    if imported_model_info.voiceChangerType == "RVC":
        imported_model_info_key.extend(
            list(RVCImportedModelInfo.__annotations__.keys())
        )
        return RVCImportedModelInfo(
            id=id,
            storageDir=storage_dir,
            **{k: v for k, v in json_dict.items() if k in imported_model_info_key},
        )
    else:
        return None


def load_all_imported_model_infos(model_dir: str) -> dict[int, ImportedModelInfo]:
    imported_model_infos: dict[ImportedModelInfo] = {}
    if not os.path.exists(model_dir):
        return imported_model_infos
    for subdir in [d for d in os.listdir(model_dir) if d.isdigit()]:
        id = int(subdir)
        imported_model_info = load_imported_model_info(
            id, os.path.join(model_dir, subdir)
        )
        if imported_model_info is not None:
            imported_model_infos[id] = imported_model_info
    return imported_model_infos


def save_imported_model_info(imported_model_info: ImportedModelInfo):
    storage_dir = imported_model_info.storageDir
    os.makedirs(storage_dir, exist_ok=True)
    logger.info(f"ImportedModelInfo::: {imported_model_info}")
    imported_model_info_dict = asdict(imported_model_info)
    del imported_model_info_dict["id"]
    del imported_model_info_dict["storageDir"]
    with open(os.path.join(storage_dir, "params.json"), "w") as f:
        json.dump(imported_model_info_dict, f, indent=4)
