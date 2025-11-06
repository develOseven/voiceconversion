import logging
import os

import torch

from voiceconversion.const import PTH_MERGED_FILENAME, TMP_DIR
from voiceconversion.imported_model_info_manager import ModelSlotManager
from voiceconversion.RVC.model_merger.merge_model import merge_model
from voiceconversion.utils.ModelMerger import ModelMerger, ModelMergerRequest

logger = logging.getLogger(__name__)


class RVCModelMerger(ModelMerger):
    @classmethod
    def merge_models(
        cls,
        slot_manager: ModelSlotManager,
        request: ModelMergerRequest,
        store_slot: int,
    ) -> str:
        model = merge_model(slot_manager, request)

        # いったんは、アップロードフォルダに格納する。（歴史的経緯）
        # 後続のloadmodelを呼び出すことで永続化モデルフォルダに移動させられる。
        logger.info(f"store merged model to: {TMP_DIR}")
        os.makedirs(TMP_DIR, exist_ok=True)
        merged_file = os.path.join(TMP_DIR, PTH_MERGED_FILENAME)
        # Save as PTH for compatibility with other implementations
        torch.save(model, merged_file)
        return merged_file
