from __future__ import annotations

import asyncio
import math
from typing import Callable, List, Optional

import msgpack
import numpy as np
import websockets
from loguru import logger

from capy.utils.worker import AsyncWorker


# The Kyutai STT rust server expects mono float32 PCM at 24kHz.
SAMPLE_RATE = 24000
ASR_STREAMING_PATH = "/api/asr-streaming"

# Kyutai STT processes audio in fixed-size frames. Unmute sends 1920 samples (80 ms) per frame.
SAMPLES_PER_FRAME = 1920
FRAME_TIME_SEC = SAMPLES_PER_FRAME / SAMPLE_RATE

# Algorithmic delay of the STT model (0.5 s for stt-1b, 2.5 s for stt-2.6b). When the VAD
# predicts end-of-turn, words still in this buffer have not been emitted yet; the flush trick
# (sending silence frames) pushes them through before we publish the segment.
STT_DELAY_SEC = 0.5

# The semantic VAD Step message exposes pause-prediction scores in ``prs``. Unmute uses index 2.
PAUSE_PREDICTION_HEAD_INDEX = 2
PAUSE_THRESHOLD = 0.6

# Pause scores are noisy for the first few Step messages after connect.
N_STEPS_TO_IGNORE = 12


class ExponentialMovingAverage:
    """Smooth pause-prediction scores to avoid false end-of-turn on brief gaps."""

    def __init__(
        self,
        attack_time: float = 0.01,
        release_time: float = 0.01,
        initial_value: float = 1.0,
    ):
        self.value = initial_value
        self.attack_time = attack_time
        self.release_time = release_time

    def update(self, dt: float, new_value: float) -> None:
        if new_value > self.value:
            alpha = 1 - math.exp(-dt / self.attack_time)
        else:
            alpha = 1 - math.exp(-dt / self.release_time)
        self.value = alpha * new_value + (1 - alpha) * self.value


class CapyASR(AsyncWorker):
    """
    Capy's Automatic Speech Recognition (ASR) engine, backed by the Kyutai STT rust server.

    This is an :class:`AsyncWorker`: raw audio chunks are fed in through
    ``consume_nonblocking`` (e.g. by a microphone producer) and completed transcript segments
    are published onto ``output_queue``.

    Audio is streamed to the rust server over a WebSocket and transcribed there. The model
    ships its own semantic VAD, so segmentation is delegated to the server: words are received
    incrementally and a segment is emitted after the VAD predicts an end-of-turn pause and the
    model delay buffer has been flushed (see Kyutai's "flush trick").

    Protocol (msgpack over WebSocket, see the Kyutai delayed-streams-modeling samples):
    - Send: ``{"type": "Audio", "pcm": [float, ...]}`` with mono float32 samples in [-1, 1].
    - Receive: ``{"type": "Word", "text": str}`` and ``{"type": "Step", "prs": [float, ...]}``.

    Expected input: mono float32 PCM chunks (``np.ndarray``) sampled at ``SAMPLE_RATE``.
    """

    def __init__(
        self,
        url: str = "ws://127.0.0.1:8080",
        api_key: str = "public_token",
        pause_prediction_head_index: int = PAUSE_PREDICTION_HEAD_INDEX,
        pause_threshold: float = PAUSE_THRESHOLD,
        delay_sec: float = STT_DELAY_SEC,
    ):
        """
        Initialize the ASR engine.

        Args:
            url (str): Base URL of the Kyutai STT rust server.
            api_key (str): API key sent via the ``kyutai-api-key`` header.
            pause_prediction_head_index (int): Index into the Step message ``prs`` array for
                pause prediction (Unmute uses 2).
            pause_threshold (float): Smoothed pause score above which end-of-turn is detected.
            delay_sec (float): STT model algorithmic delay; used by the flush trick after VAD.
        """
        super().__init__()

        self.ws_url = url.rstrip("/") + ASR_STREAMING_PATH
        self.api_key = api_key
        self.pause_prediction_head_index = pause_prediction_head_index
        self.pause_threshold = pause_threshold
        self.delay_sec = delay_sec

        # Completed transcript segments are published here for downstream consumers.
        self.output_queue: asyncio.Queue[str] = asyncio.Queue()

        # Words accumulated for the current (in-progress) segment.
        self._words: List[str] = []
        # Whether speech has been seen since the last emitted segment.
        self._speech_started = False
        self._cli_transcription_consumer: Optional[Callable[[str], None]] = None

        # Tracks model time via Step messages; starts negative by one model delay.
        self._current_time = -self.delay_sec
        self._pause_prediction = ExponentialMovingAverage()
        self._steps_to_ignore = N_STEPS_TO_IGNORE
        self._flushing = False
        self._flush_end_time: Optional[float] = None
        self._flush_queue: asyncio.Queue[np.ndarray] = asyncio.Queue()

    def set_cli_output(self, cli_output: object | None) -> None:
        """Attach a CLI output worker to receive completed transcript segments."""
        if cli_output is None:
            self._cli_transcription_consumer = None
            return
        self._cli_transcription_consumer = cli_output.consume_transcription

    def _publish_transcript(self, transcript: str) -> None:
        self.output_queue.put_nowait(transcript)
        if self._cli_transcription_consumer is not None:
            self._cli_transcription_consumer(transcript)

    def _start_flush(self) -> None:
        """Push silence through the STT delay buffer so trailing words are emitted."""
        self._flushing = True
        self._flush_end_time = self._current_time + self.delay_sec
        num_frames = int(math.ceil(self.delay_sec / FRAME_TIME_SEC)) + 1
        zero_frame = np.zeros(SAMPLES_PER_FRAME, dtype=np.float32)
        for _ in range(num_frames):
            self._flush_queue.put_nowait(zero_frame)

    def _finish_segment(self) -> None:
        """Publish the accumulated segment and reset turn state."""
        transcript = " ".join(self._words)
        if transcript.strip():
            self._publish_transcript(transcript)
        self._words = []
        self._speech_started = False
        self._flushing = False
        self._flush_end_time = None

    async def _send_audio(self, websocket):
        """Drain audio chunks from the input queue and stream them to the server."""
        # Drop any audio buffered before the connection was ready to avoid lag.
        while not self.input_queue.empty():
            self.input_queue.get_nowait()

        while True:
            if not self._flush_queue.empty():
                pcm = self._flush_queue.get_nowait()
            else:
                chunk = await self.input_queue.get()
                pcm = np.asarray(chunk, dtype=np.float32).reshape(-1)

            msg = msgpack.packb(
                {"type": "Audio", "pcm": pcm.tolist()},
                use_bin_type=True,
                use_single_float=True,
            )
            await websocket.send(msg)

    def _handle_message(self, data: dict):
        """Update transcript state from a single decoded server message."""
        msg_type = data.get("type")
        if msg_type == "Word":
            # First word of a new turn: interrupt the rest of the pipeline so the assistant
            # stops talking as soon as the user starts speaking (before end-of-turn).
            if not self._speech_started:
                self.broadcast_interrupt()
            self._words.append(data["text"])
            self._speech_started = True
        elif msg_type == "Step":
            self._current_time += FRAME_TIME_SEC
            if self._steps_to_ignore > 0:
                self._steps_to_ignore -= 1
            else:
                pause_prediction = data["prs"][self.pause_prediction_head_index]
                self._pause_prediction.update(FRAME_TIME_SEC, pause_prediction)

            if (
                not self._flushing
                and self._speech_started
                and self._pause_prediction.value > self.pause_threshold
            ):
                # End-of-turn detected: flush the delay buffer before emitting the segment.
                self._start_flush()
            elif (
                self._flushing
                and self._flush_end_time is not None
                and self._current_time > self._flush_end_time
                and self._flush_queue.empty()
            ):
                self._finish_segment()
        elif msg_type == "Error":
            logger.error("Kyutai STT server error: {}", data.get("message"))

    async def _receive_transcripts(self, websocket):
        """Receive server messages, accumulate words, and emit VAD-segmented transcripts."""
        async for message in websocket:
            self._handle_message(msgpack.unpackb(message, raw=False))

    async def _run_loop(self):
        """
        Connect to the Kyutai STT rust server and run the send/receive streaming loop until
        the worker task is cancelled.
        """
        headers = {"kyutai-api-key": self.api_key}
        async with websockets.connect(self.ws_url, additional_headers=headers) as websocket:
            send_task = asyncio.create_task(self._send_audio(websocket))
            receive_task = asyncio.create_task(self._receive_transcripts(websocket))
            try:
                await asyncio.gather(send_task, receive_task)
            finally:
                send_task.cancel()
                receive_task.cancel()

    async def get_transcript(self) -> str:
        """Await the next completed transcript segment."""
        return await self.output_queue.get()
