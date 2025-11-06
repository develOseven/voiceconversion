import json
import logging
import os
from dataclasses import asdict

import onnxruntime
import safetensors
import torch

from voiceconversion.common.SafetensorsUtils import convert_single
from voiceconversion.const import EnumInferenceTypes
from voiceconversion.data.imported_model_info import (
    ImportedModelInfo,
    RVCImportedModelInfo,
)
from voiceconversion.utils.import_model_params import ImportModelParams
from voiceconversion.utils.model_importer import ModelImporter

logger = logging.getLogger(__name__)


class RVCModelImporter(ModelImporter):
    @classmethod
    def import_model(
        cls,
        id: int,
        storage_dir: str,
        props: ImportModelParams,
    ) -> ImportedModelInfo:
        imported_model_info: RVCImportedModelInfo = RVCImportedModelInfo()
        imported_model_info.id = id
        imported_model_info.storageDir = storage_dir
        for file in props.files:
            if file.kind == "rvcModel":
                imported_model_info.modelFile = file.name
            elif file.kind == "rvcIndex":
                imported_model_info.indexFile = file.name
        imported_model_info.defaultTune = 0
        imported_model_info.defaultFormantShift = 0
        imported_model_info.defaultIndexRatio = 0
        imported_model_info.defaultProtect = 0.5
        imported_model_info.isONNX = imported_model_info.modelFile.endswith(".onnx")
        imported_model_info.name = os.path.splitext(
            os.path.basename(imported_model_info.modelFile)
        )[0]
        logger.info(f"RVC:: modelFile {imported_model_info.modelFile}")

        model_path = os.path.join(
            imported_model_info.storageDir,
            os.path.basename(imported_model_info.modelFile),
        )
        if imported_model_info.isONNX:
            imported_model_info = cls._set_info_by_onnx(model_path, imported_model_info)
        else:
            imported_model_info = cls._set_info_by_pytorch(
                model_path, imported_model_info
            )
            if not model_path.endswith(".safetensors"):
                convert_single(model_path, True)
                filename, _ = os.path.splitext(os.path.basename(model_path))
                imported_model_info.modelFile = f"{filename}.safetensors"
        return imported_model_info

    @classmethod
    def _set_info_by_pytorch(
        cls, model_path: str, imported_model_info: RVCImportedModelInfo
    ):
        if model_path.endswith(".safetensors"):
            with safetensors.safe_open(model_path, "pt") as data:
                cpt = data.metadata()
                cpt["f0"] = int(cpt["f0"])
                cpt["config"] = json.loads(cpt["config"])
        else:
            cpt = torch.load(model_path, map_location="cpu")
        config_len = len(cpt["config"])
        version = cpt.get("version", "v1")

        imported_model_info = RVCImportedModelInfo(**asdict(imported_model_info))
        imported_model_info.f0 = True if cpt["f0"] == 1 else False

        if config_len == 18:
            embedder = cpt.get("embedder_model", "hubert_base").replace("-", "_")
            # Original RVC
            if version == "v1":
                imported_model_info.modelType = (
                    EnumInferenceTypes.pyTorchRVC.value
                    if imported_model_info.f0
                    else EnumInferenceTypes.pyTorchRVCNono.value
                )
                imported_model_info.embChannels = 256
                imported_model_info.embOutputLayer = 9
                imported_model_info.useFinalProj = True
                imported_model_info.embedder = embedder
                logger.info("Official Model(pyTorch) : v1")
            else:
                imported_model_info.modelType = (
                    EnumInferenceTypes.pyTorchRVCv2.value
                    if imported_model_info.f0
                    else EnumInferenceTypes.pyTorchRVCv2Nono.value
                )
                imported_model_info.embChannels = 768
                imported_model_info.embOutputLayer = 12
                imported_model_info.useFinalProj = False
                imported_model_info.embedder = embedder
                logger.info("Official Model(pyTorch) : v2")

        else:
            # DDPN RVC
            imported_model_info.f0 = True if cpt["f0"] == 1 else False
            imported_model_info.modelType = (
                EnumInferenceTypes.pyTorchWebUI.value
                if imported_model_info.f0
                else EnumInferenceTypes.pyTorchWebUINono.value
            )
            imported_model_info.embChannels = cpt["config"][17]
            imported_model_info.embOutputLayer = (
                cpt["embedder_output_layer"] if "embedder_output_layer" in cpt else 9
            )
            if imported_model_info.embChannels == 256:
                imported_model_info.useFinalProj = True
            else:
                imported_model_info.useFinalProj = False

            # DDPNモデルの情報を表示
            if (
                imported_model_info.embChannels == 256
                and imported_model_info.embOutputLayer == 9
                and imported_model_info.useFinalProj
            ):
                logger.info("DDPN Model(pyTorch) : Official v1 like")
            elif (
                imported_model_info.embChannels == 768
                and imported_model_info.embOutputLayer == 12
                and imported_model_info.useFinalProj is False
            ):
                logger.info("DDPN Model(pyTorch): Official v2 like")
            else:
                logger.info(
                    f"DDPN Model(pyTorch): ch:{imported_model_info.embChannels}, L:{imported_model_info.embOutputLayer}, FP:{imported_model_info.useFinalProj}"
                )

            imported_model_info.embedder = cpt["embedder_name"]
            if imported_model_info.embedder.endswith("768"):
                imported_model_info.embedder = imported_model_info.embedder[:-3]

            if "speaker_info" in cpt.keys():
                for k, v in cpt["speaker_info"].items():
                    imported_model_info.speakers[int(k)] = str(v)

        imported_model_info.samplingRate = cpt["config"][-1]

        del cpt

        return imported_model_info

    @classmethod
    def _set_info_by_onnx(
        cls, model_path: str, imported_model_info: RVCImportedModelInfo
    ):
        tmp_onnx_session = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        modelmeta = tmp_onnx_session.get_modelmeta()
        try:
            imported_model_info = RVCImportedModelInfo(**asdict(imported_model_info))
            metadata = json.loads(modelmeta.custom_metadata_map["metadata"])

            # slot.modelType = metadata["modelType"]
            imported_model_info.embChannels = metadata["embChannels"]

            imported_model_info.embOutputLayer = (
                metadata["embOutputLayer"] if "embOutputLayer" in metadata else 9
            )
            imported_model_info.useFinalProj = (
                metadata["useFinalProj"]
                if "useFinalProj" in metadata
                else True if imported_model_info.embChannels == 256 else False
            )

            if imported_model_info.embChannels == 256:
                imported_model_info.useFinalProj = True
            else:
                imported_model_info.useFinalProj = False

            # ONNXモデルの情報を表示
            if (
                imported_model_info.embChannels == 256
                and imported_model_info.embOutputLayer == 9
                and imported_model_info.useFinalProj
            ):
                logger.info("ONNX Model: Official v1 like")
            elif (
                imported_model_info.embChannels == 768
                and imported_model_info.embOutputLayer == 12
                and imported_model_info.useFinalProj is False
            ):
                logger.info("ONNX Model: Official v2 like")
            else:
                logger.info(
                    f"ONNX Model: ch:{imported_model_info.embChannels}, L:{imported_model_info.embOutputLayer}, FP:{imported_model_info.useFinalProj}"
                )

            if "embedder" not in metadata:
                imported_model_info.embedder = "hubert_base"
            else:
                imported_model_info.embedder = metadata["embedder"]

            imported_model_info.f0 = metadata["f0"]
            imported_model_info.modelType = (
                EnumInferenceTypes.onnxRVC.value
                if imported_model_info.f0
                else EnumInferenceTypes.onnxRVCNono.value
            )
            imported_model_info.samplingRate = metadata["samplingRate"]
            imported_model_info.deprecated = False

            if imported_model_info.embChannels == 256:
                if metadata["version"] == "2.1":
                    imported_model_info.version = (
                        "v1.1"  # 1.1はclipをonnx内部で実施. realtimeをdisable
                    )
                else:
                    imported_model_info.version = "v1"
            elif metadata["version"] == "2":
                imported_model_info.version = "v2"
            elif (
                metadata["version"] == "2.1"
            ):  # 2.1はclipをonnx内部で実施. realtimeをdisable
                imported_model_info.version = "v2.1"
            elif metadata["version"] == "2.2":  # 2.1と同じ
                imported_model_info.version = "v2.2"
        except Exception as e:
            imported_model_info.modelType = EnumInferenceTypes.onnxRVC.value
            imported_model_info.embChannels = 256
            imported_model_info.embedder = "hubert_base"
            imported_model_info.f0 = True
            imported_model_info.samplingRate = 48000
            imported_model_info.deprecated = True

            logger.error("setInfoByONNX", e)
            logger.error("############## !!!! CAUTION !!!! ####################")
            logger.error("This onnxfie is deprecated. Please regenerate onnxfile.")
            logger.error("############## !!!! CAUTION !!!! ####################")

        del tmp_onnx_session
        return imported_model_info
