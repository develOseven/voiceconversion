import logging
import os
import shutil
from typing import Tuple

from voiceconversion.data.imported_model_info import (
    ImportedModelInfo,
    load_all_imported_model_infos,
    save_imported_model_info,
)

logger = logging.getLogger(__name__)


class ImportedModelInfoManager:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.infos = load_all_imported_model_infos(self.model_dir)

    def save(self, info: ImportedModelInfo):
        save_imported_model_info(info)
        self.infos = load_all_imported_model_infos(self.model_dir)

    def get(self, id: int) -> ImportedModelInfo | None:
        return self.infos.get(id)

    def new_id(self) -> Tuple[int, str]:
        if self.infos:
            id = max(self.infos) + 1
        else:
            id = 0
        return id, os.path.join(self.model_dir, str(id))

    def remove(self, id: int):
        imported_model_info = self.get(id)
        if imported_model_info is None:
            return
        if os.path.exists(imported_model_info.storageDir):
            shutil.rmtree(imported_model_info.storageDir)
        del self.infos[id]
