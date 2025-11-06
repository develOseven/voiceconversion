import logging
import os
import shutil

from voiceconversion.data.imported_model_info import ImportedModelInfo
from voiceconversion.imported_model_info_manager import ImportedModelInfoManager
from voiceconversion.RVC.rvc_model_importer import RVCModelImporter
from voiceconversion.utils.import_model_params import ImportModelParams

logger = logging.getLogger(__name__)


def import_model(
    imported_model_info_manager: ImportedModelInfoManager,
    params: ImportModelParams,
    imported_model_info: ImportedModelInfo | None,
) -> ImportedModelInfo | None:
    if imported_model_info is None:
        id, storage_dir = imported_model_info_manager.new_id()
    else:
        id = imported_model_info.id
        storage_dir = imported_model_info.storageDir

    if os.path.isdir(storage_dir):
        # Replacing existing model, delete everything.
        shutil.rmtree(storage_dir)

    for file in params.files:
        logger.info(f"FILE: {file}")
        src_path = os.path.join(file.dir, file.name)
        dst_dir = os.path.join(
            storage_dir,
            file.dir,
        )
        dst_path = os.path.join(dst_dir, os.path.basename(file.name))
        os.makedirs(dst_dir, exist_ok=True)
        logger.info(f"Copying {src_path} -> {dst_path}")
        shutil.copy(src_path, dst_path)
        file.name = os.path.basename(dst_path)

    if params.voice_changer_type == "RVC":
        imported_model_info = RVCModelImporter.import_model(id, storage_dir, params)
        imported_model_info_manager.save(imported_model_info)
        return imported_model_info

    return None
