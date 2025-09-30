from typing import Protocol

from voiceconversion.utils.LoadModelParams import LoadModelParams


class ModelSlotGenerator(Protocol):
    @classmethod
    def load_model(cls, params: LoadModelParams):
        ...
