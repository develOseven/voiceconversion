import logging
from typing import Any, Union

import numpy as np
import torch
from torch.functional import F

from voiceconversion.common.deviceManager.DeviceManager import DeviceManager
from voiceconversion.const import VoiceChangerType
from voiceconversion.data.imported_model_info import ImportedModelInfo
from voiceconversion.Exceptions import VoiceChangerIsNotSelectedException
from voiceconversion.IORecorder import IORecorder
from voiceconversion.utils.Timer import Timer2
from voiceconversion.utils.VoiceChangerModel import AudioInOutFloat, VoiceChangerModel
from voiceconversion.voice_changer_settings import VoiceChangerSettings

logger = logging.getLogger(__name__)


class VoiceChangerV2:
    def __init__(
        self, settings: VoiceChangerSettings, io_recorder_dir: str | None = None
    ):
        self.settings = settings

        self.block_frame = self.settings.serverReadChunkSize * 128
        self.crossfade_frame = int(
            self.settings.crossFadeOverlapSize * self.settings.inputSampleRate
        )
        self.extra_frame = int(
            self.settings.extraConvertSize * self.settings.inputSampleRate
        )
        self.sola_search_frame = self.settings.inputSampleRate // 100

        self.vcmodel: VoiceChangerModel | None = None
        self.device = DeviceManager.get_instance().device
        self.sola_buffer: torch.Tensor | None = None
        if io_recorder_dir is not None:
            self.io_recorder: IORecorder | None = IORecorder(
                self.settings.inputSampleRate,
                self.settings.outputSampleRate,
                io_recorder_dir,
            )
        else:
            self.io_recorder = None
        self._generate_strength()

    def initialize(self, vcmodel: VoiceChangerModel, pretrain_dir: str):
        self.vcmodel = vcmodel
        self.vcmodel.realloc(
            self.block_frame,
            self.extra_frame,
            self.crossfade_frame,
            self.sola_search_frame,
        )
        self.vcmodel.initialize(force_reload=False, pretrain_dir=pretrain_dir)

    def _generate_strength(self) -> None:
        self.fade_in_window: torch.Tensor = (
            torch.sin(
                0.5
                * np.pi
                * torch.linspace(
                    0.0,
                    1.0,
                    steps=self.crossfade_frame,
                    device=self.device,
                    dtype=torch.float32,
                )
            )
            ** 2
        )
        self.fade_out_window: torch.Tensor = 1 - self.fade_in_window

        # ひとつ前の結果とサイズが変わるため、記録は消去する。
        self.sola_buffer = torch.zeros(
            self.crossfade_frame, device=self.device, dtype=torch.float32
        )
        logger.info(f"Allocated SOLA buffer size: {self.crossfade_frame}")

    def process_audio(self, audio_in: AudioInOutFloat) -> tuple[AudioInOutFloat, float]:
        assert self.vcmodel is not None
        assert self.sola_buffer is not None

        block_size = audio_in.shape[0]

        audio, vol = self.vcmodel.inference(audio_in)

        if audio is None:
            # In case there's an actual silence - send full block with zeros
            return np.zeros(block_size, dtype=np.float32), vol

        # SOLA algorithm from https://github.com/yxlllc/DDSP-SVC, https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI
        conv_input = audio[None, None, : self.crossfade_frame + self.sola_search_frame]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(
                conv_input**2,
                torch.ones(1, 1, self.crossfade_frame, device=self.device),
            )
            + 1e-8
        )
        sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])

        audio = audio[sola_offset:]
        audio[: self.crossfade_frame] *= self.fade_in_window
        audio[: self.crossfade_frame] += self.sola_buffer * self.fade_out_window

        self.sola_buffer[:] = audio[block_size : block_size + self.crossfade_frame]

        return audio[:block_size].detach().cpu().numpy(), vol

    @torch.no_grad()
    def on_request(
        self, audio_in: AudioInOutFloat
    ) -> tuple[AudioInOutFloat, float, list[Union[int, float]]]:
        if self.vcmodel is None:
            raise VoiceChangerIsNotSelectedException("Voice Changer is not selected.")

        with Timer2("main-process", True) as t:
            result, vol = self.process_audio(audio_in)

        mainprocess_time = t.secs

        # Post-processing
        if self.io_recorder is not None and self.settings.recordIO:
            self.io_recorder.write_input((audio_in * 32767).astype(np.int16).tobytes())
            self.io_recorder.write_output((result * 32767).astype(np.int16).tobytes())

        return result, vol, [0, mainprocess_time, 0]

    @torch.no_grad()
    def export2onnx(self):
        return self.vcmodel.export2onnx()
