from const import EnumInferenceTypes
from voiceconversion.RVC.inferencer.Inferencer import Inferencer
from voiceconversion.RVC.inferencer.OnnxRVCInferencer import OnnxRVCInferencer
from voiceconversion.RVC.inferencer.OnnxRVCInferencerNono import OnnxRVCInferencerNono
from voiceconversion.RVC.inferencer.RVCInferencer import RVCInferencer
from voiceconversion.RVC.inferencer.RVCInferencerNono import RVCInferencerNono
from voiceconversion.RVC.inferencer.RVCInferencerv2 import RVCInferencerv2
from voiceconversion.RVC.inferencer.RVCInferencerv2Nono import RVCInferencerv2Nono
from voiceconversion.RVC.inferencer.WebUIInferencer import WebUIInferencer
from voiceconversion.RVC.inferencer.WebUIInferencerNono import WebUIInferencerNono


class InferencerManager:
    currentInferencer: Inferencer | None = None

    @classmethod
    def getInferencer(
        cls,
        inferencerType: str,
        file: str,
    ) -> Inferencer:
        cls.currentInferencer = cls.loadInferencer(EnumInferenceTypes(inferencerType), file)
        return cls.currentInferencer

    @classmethod
    def loadInferencer(
        cls,
        inferencerType: EnumInferenceTypes,
        file: str,
    ) -> Inferencer:
        if inferencerType is EnumInferenceTypes.pyTorchRVC:
            return RVCInferencer().load_model(file)
        elif inferencerType is EnumInferenceTypes.pyTorchRVCNono:
            return RVCInferencerNono().load_model(file)
        elif inferencerType == EnumInferenceTypes.pyTorchRVCv2:
            return RVCInferencerv2().load_model(file)
        elif inferencerType is EnumInferenceTypes.pyTorchRVCv2Nono:
            return RVCInferencerv2Nono().load_model(file)
        elif inferencerType is EnumInferenceTypes.pyTorchWebUI:
            return WebUIInferencer().load_model(file)
        elif inferencerType is EnumInferenceTypes.pyTorchWebUINono:
            return WebUIInferencerNono().load_model(file)
        elif inferencerType is EnumInferenceTypes.onnxRVC:
            return OnnxRVCInferencer().load_model(file)
        elif inferencerType is EnumInferenceTypes.onnxRVCNono:
            return OnnxRVCInferencerNono().load_model(file)
        else:
            raise RuntimeError("Inferencer not found", inferencerType)
