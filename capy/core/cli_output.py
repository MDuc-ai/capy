from __future__ import annotations

from dataclasses import dataclass

from capy.utils.worker import InterruptibleAsyncWorker, InterruptibleEvent


@dataclass(frozen=True)
class TranscriptionDisplay:
    text: str


@dataclass(frozen=True)
class LlmResponseEnd:
    pass


class CapyCliOutput(InterruptibleAsyncWorker):
    """Print completed speech transcriptions and streaming LLM response chunks."""

    def __init__(self) -> None:
        super().__init__()
        self._printing_assistant = False

    def consume_transcription(self, transcription: str) -> None:
        self.consume_nonblocking(
            InterruptibleEvent(
                data=TranscriptionDisplay(transcription),
                is_interruptible=False,
            )
        )

    def consume_llm_chunk(self, chunk: str) -> None:
        self.consume_nonblocking(
            InterruptibleEvent(data=chunk, is_interruptible=False)
        )

    def finish_llm_response(self) -> None:
        self.consume_nonblocking(
            InterruptibleEvent(data=LlmResponseEnd(), is_interruptible=False)
        )

    async def process(self, item: InterruptibleEvent) -> None:
        data = item.data
        if isinstance(data, TranscriptionDisplay):
            print(f"\033[92mUser: \033[0m{data.text}")
            return

        if isinstance(data, LlmResponseEnd):
            if self._printing_assistant:
                print()
                self._printing_assistant = False
            return

        if isinstance(data, str):
            if not self._printing_assistant:
                print("\033[93mAssistant: \033[0m", end="", flush=True)
                self._printing_assistant = True
            print(data, end="", flush=True)
