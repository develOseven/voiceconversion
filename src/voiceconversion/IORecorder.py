import wave
import os
import logging

logger = logging.getLogger(__name__)


class IORecorder:
    def __init__(self, input_sampling_rate: int, output_sampling_rate: int, tmp_dir: str):
        self.fi = None
        self.fo = None
        self.stream_input_file = os.path.join(tmp_dir, "in.wav")
        self.stream_output_file = os.path.join(tmp_dir, "out.wav")
        self.open(input_sampling_rate, output_sampling_rate)

    def _clear(self):
        self.close()
        self._clearFile(self.stream_input_file)
        self._clearFile(self.stream_output_file)

    def _clearFile(self, filename: str):
        if os.path.exists(filename):
            logger.info(f"Removing old recording file {filename}")
            os.remove(filename)

    def open(self, input_sampling_rate: int, output_sampling_rate: int):
        self._clear()

        self.fi = wave.open(self.stream_input_file, "wb")
        self.fi.setnchannels(1)
        self.fi.setsampwidth(2)
        self.fi.setframerate(input_sampling_rate)

        self.fo = wave.open(self.stream_output_file, "wb")
        self.fo.setnchannels(1)
        self.fo.setsampwidth(2)
        self.fo.setframerate(output_sampling_rate)
        logger.info(f"-------------------------- - - - {self.stream_input_file}, {self.stream_output_file}")

    def write_input(self, wav):
        if self.fi is None:
            raise Exception('IO recorder is closed.')
        self.fi.writeframes(wav)

    def write_output(self, wav):
        if self.fo is None:
            raise Exception('IO recorder is closed.')
        self.fo.writeframes(wav)

    def close(self):
        if self.fi is not None:
            self.fi.close()
            self.fi = None
        if self.fo is not None:
            self.fo.close()
            self.fo = None
