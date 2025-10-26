import json
import logging
import os
import shutil
from voiceconversion.data.ModelSlot import ModelSlots, loadAllSlotInfo, loadSlotInfo, saveSlotInfo

logger = logging.getLogger(__name__)


class ModelSlotManager:
    _instance = None

    def __init__(self, model_dir: str, upload_dir: str):
        self.model_dir = model_dir
        self.upload_dir = upload_dir
        self.modelSlots = loadAllSlotInfo(self.model_dir)

    @classmethod
    def get_instance(cls, model_dir: str, upload_dir: str):
        if cls._instance is None:
            cls._instance = cls(model_dir, upload_dir)
        return cls._instance

    def _save_model_slot(self, slotIndex: int, slotInfo: ModelSlots):
        saveSlotInfo(self.model_dir, slotIndex, slotInfo)
        self.modelSlots = loadAllSlotInfo(self.model_dir)

    def _load_model_slot(self, slotIndex: int):
        return self.modelSlots[slotIndex]

    def getAllSlotInfo(self, reload: bool = False):
        if reload:
            self.modelSlots = loadAllSlotInfo(self.model_dir)
        return self.modelSlots

    def get_slot_info(self, slotIndex: int):
        if slotIndex == -1:
            return
        return self._load_model_slot(slotIndex)

    def save_model_slot(self, slotIndex: int, slotInfo: ModelSlots):
        self._save_model_slot(slotIndex, slotInfo)

    def update_model_info(self, slot_index: int, key: str, val):
        logger.info(f"UPDATE MODEL INFO: {key}={val}")
        slotInfo = self._load_model_slot(slot_index)
        if key == "speakers":
            setattr(slotInfo, key, json.loads(val))
        else:
            setattr(slotInfo, key, val)
        self._save_model_slot(slot_index, slotInfo)

    def store_model_assets(self, params: str):
        paramsDict = json.loads(params)
        uploadPath = os.path.join(self.upload_dir, paramsDict["file"])
        storeDir = os.path.join(self.model_dir, str(paramsDict["slot"]))
        storePath = os.path.join(
            storeDir,
            paramsDict["file"],
        )
        try:
            shutil.move(uploadPath, storePath)
            slotInfo = self._load_model_slot(paramsDict["slot"])
            setattr(slotInfo, paramsDict["name"], storePath)
            self._save_model_slot(paramsDict["slot"], slotInfo)
        except Exception as e:
            logger.exception(e)

    def renumberSlots(
        self,
        sourceStart: int,
        sourceEnd: int,
        destinationRow: int,
    ):
        if destinationRow > sourceEnd:
            blockSize = sourceEnd - sourceStart + 1
            for slotInfo in self.modelSlots:
                oldIndex = slotInfo.slotIndex

                if sourceStart <= oldIndex <= sourceEnd:
                    newIndex = oldIndex + (destinationRow - sourceEnd - 1)
                elif sourceEnd < oldIndex < destinationRow:
                    newIndex = oldIndex - blockSize
                else:
                    newIndex = oldIndex

                if newIndex != oldIndex:
                    slotInfo.slotIndex = newIndex
                    saveSlotInfo(self.model_dir, newIndex, slotInfo)

        elif destinationRow < sourceStart:
            blockSize = sourceEnd - sourceStart + 1
            for slotInfo in self.modelSlots:
                oldIndex = slotInfo.slotIndex

                if sourceStart <= oldIndex <= sourceEnd:
                    newIndex = oldIndex - (sourceStart - destinationRow)
                elif destinationRow <= oldIndex < sourceStart:
                    newIndex = oldIndex + blockSize
                else:
                    newIndex = oldIndex

                if newIndex != oldIndex:
                    slotInfo.slotIndex = newIndex
                    saveSlotInfo(self.model_dir, newIndex, slotInfo)

        self.modelSlots = sorted(
            self.modelSlots,
            key=lambda slotInfo: slotInfo.slotIndex,
        )

    def removeSlots(self, first: int, last: int):
        count = last - first + 1

        remaining = []
        for slotInfo in self.modelSlots:
            oldIndex = slotInfo.slotIndex
            if first <= oldIndex <= last:
                continue
            remaining.append(slotInfo)
        self.modelSlots = remaining

        for slotInfo in self.modelSlots:
            oldIndex = slotInfo.slotIndex
            if oldIndex > last:
                newIndex = oldIndex - count
                slotInfo.slotIndex = newIndex
                saveSlotInfo(self.model_dir, newIndex, slotInfo)
