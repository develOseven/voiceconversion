from typing import Protocol

from voiceconversion.data.imported_model_info import ImportedModelInfo
from voiceconversion.utils.import_model_params import ImportModelParams


class ModelImporter(Protocol):
    @classmethod
    def import_model(
        cls,
        id: int,
        storage_dir: str,
        params: ImportModelParams,
    ) -> ImportedModelInfo: ...
