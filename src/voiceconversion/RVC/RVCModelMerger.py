import os
import torch
import logging
from const import TMP_DIR, PTH_MERGED_FILENAME
from voiceconversion.RVC.model_merger.MergeModel import merge_model
from voiceconversion.utils.ModelMerger import ModelMerger, ModelMergerRequest


logger = logging.getLogger(__name__)


class RVCModelMerger(ModelMerger):
    @classmethod
    def merge_models(cls, model_dir: str, request: ModelMergerRequest, store_slot: int) -> str:
        model = merge_model(model_dir, request)

        # いったんは、アップロードフォルダに格納する。（歴史的経緯）
        # 後続のloadmodelを呼び出すことで永続化モデルフォルダに移動させられる。
        logger.info(f"store merged model to: {TMP_DIR}")
        os.makedirs(TMP_DIR, exist_ok=True)
        merged_file = os.path.join(TMP_DIR, PTH_MERGED_FILENAME)
        # Save as PTH for compatibility with other implementations
        torch.save(model, merged_file)
        return merged_file
