from __future__ import annotations

import asyncio
from typing import Optional

import numpy as np
import sounddevice as sd


class MicrophoneInput:
    DEFAULT_SAMPLING_RATE = 44100
    DEFAULT_CHUNK_SIZE = 2048

    def __init__(
        self,
        device_info: dict,
        sampling_rate: Optional[int] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        microphone_gain: int = 1,
    ):
        self.device_info = device_info
        self.sampling_rate = int(
            sampling_rate or self.device_info.get("default_samplerate", self.DEFAULT_SAMPLING_RATE)
        )
        self.chunk_size = chunk_size
        self.microphone_gain = microphone_gain
        self._loop = asyncio.get_running_loop()
        self.output_queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self.stream = sd.InputStream(
            dtype=np.int16,
            channels=1,
            samplerate=self.sampling_rate,
            blocksize=self.chunk_size,
            device=int(self.device_info["index"]),
            callback=self._stream_callback,
        )
        self.stream.start()

    def _apply_gain(self, samples: np.ndarray) -> np.ndarray:
        if self.microphone_gain > 1:
            factor = 1 << self.microphone_gain
            samples = np.clip(
                samples.astype(np.int32) * factor, -32768, 32767
            ).astype(np.int16)
        elif self.microphone_gain > 0:
            samples = samples // (1 << self.microphone_gain)
        return samples

    def _stream_callback(self, in_data: np.ndarray, *_args):
        samples = self._apply_gain(in_data.reshape(-1))
        pcm_float32 = samples.astype(np.float32) / 32768.0
        self._loop.call_soon_threadsafe(self.output_queue.put_nowait, pcm_float32)

    def close(self):
        if self.stream.active:
            self.stream.stop()
        self.stream.close()

    @classmethod
    def from_default_device(
        cls,
        sampling_rate: Optional[int] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        microphone_gain: int = 1,
    ):
        device_info = sd.query_devices(kind="input")
        return cls(
            device_info,
            sampling_rate=sampling_rate,
            chunk_size=chunk_size,
            microphone_gain=microphone_gain,
        )
