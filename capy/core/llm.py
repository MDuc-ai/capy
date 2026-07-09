from __future__ import annotations

import asyncio
import threading
from typing import Callable, Literal, Optional, Union

from llama_cpp import Llama
from llama_cpp.llama_types import (
    ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestSystemMessage,
    ChatCompletionRequestUserMessage,
)
from loguru import logger

from capy.utils.worker import InterruptibleAsyncWorker, InterruptibleEvent


MODEL_PATH_MAP = {"llama-3": "./models/Llama-3.2-3B-Instruct-Q5_K_S.gguf"}


class CapyLLM(InterruptibleAsyncWorker):
    """
    Capy's LLM engine backed by llama.cpp.

    This is an :class:`InterruptibleAsyncWorker`: completed transcript segments from ASR
    are fed in through ``consume_nonblocking`` (wrapped in :class:`InterruptibleEvent`)
    and response chunks are published onto ``output_queue``.

    Expected input: :class:`InterruptibleEvent` whose ``data`` is a user transcription string.
    """

    def __init__(
        self,
        sys_prompt: str = "You are a helpful assistant",
        n_ctx: int = 4096,
        batch_size: int = 512,
    ):
        super().__init__()

        self.sys_prompt = ChatCompletionRequestSystemMessage(
            role="system", content=sys_prompt
        )
        self.n_ctx = n_ctx
        self.batch_size = batch_size
        self.messages = [self.sys_prompt]
        self.llm: Optional[Llama] = None

        self.output_queue: asyncio.Queue[str] = asyncio.Queue()
        self._stop_generation_event = threading.Event()
        self._cli_chunk_consumer: Optional[Callable[[str], None]] = None
        self._cli_response_end_consumer: Optional[Callable[[], None]] = None

    def set_cli_output(self, cli_output: object | None) -> None:
        """Attach a CLI output worker to receive streaming response chunks."""
        if cli_output is None:
            self._cli_chunk_consumer = None
            self._cli_response_end_consumer = None
            return
        self._cli_chunk_consumer = cli_output.consume_llm_chunk
        self._cli_response_end_consumer = cli_output.finish_llm_response

    def _publish_chunk(self, chunk: str) -> None:
        self.output_queue.put_nowait(chunk)
        if self._cli_chunk_consumer is not None:
            self._cli_chunk_consumer(chunk)

    def _publish_response_end(self) -> None:
        if self._cli_response_end_consumer is not None:
            self._cli_response_end_consumer()

    @property
    def is_loaded(self) -> bool:
        return self.llm is not None

    def load_llm(self, model_name_or_path: Union[str, Literal["llama-3"]] = "llama-3"):
        if model_name_or_path in MODEL_PATH_MAP:
            model_name_or_path = MODEL_PATH_MAP[model_name_or_path]
        self.llm = Llama(model_path=model_name_or_path, n_ctx=self.n_ctx, verbose=False)

    def offload_llm(self):
        if self.llm is not None:
            self.llm.close()
            self.llm = None

    def reset_llm(self):
        self.messages = []
        if self.llm is not None:
            self.llm.reset()

    def consume_transcription(self, transcription: str, is_interruptible: bool = True):
        """Enqueue a user transcription for response generation."""
        self.consume_nonblocking(
            InterruptibleEvent(data=transcription, is_interruptible=is_interruptible)
        )

    async def process(self, item: InterruptibleEvent):
        user_input = item.data
        if not isinstance(user_input, str) or not user_input.strip():
            return

        if self.llm is None:
            logger.error("No model loaded")
            return

        self.messages.append(
            ChatCompletionRequestUserMessage(role="user", content=user_input)
        )

        stop_event = threading.Event()

        # Run generation on a background thread. Shield the task so that, on barge-in,
        # cancellation does not abandon the still-running thread: llama.cpp is not
        # thread-safe, so the next turn must not call the model until this one unwinds.
        gen_task = asyncio.create_task(
            asyncio.to_thread(
                self._stream_completion,
                on_chunk=self._publish_chunk,
                stop_event=stop_event,
                interrupt_check=item.is_interrupted,
            )
        )
        try:
            await asyncio.shield(gen_task)
        except asyncio.CancelledError:
            item.interrupt()
            stop_event.set()
            # Wait for the generation thread to actually exit create_chat_completion
            # before propagating cancellation, so no two calls overlap on self.llm.
            try:
                await gen_task
            except asyncio.CancelledError:
                pass
            raise
        finally:
            self._publish_response_end()

    def _stream_completion(
        self,
        on_chunk=None,
        stop_event=None,
        interrupt_check=None,
        cli: bool = False,
    ) -> str:
        assistant_response = ""
        if cli:
            print("\033[93mAssistant: \033[0m", end="", flush=True)

        for response in self.llm.create_chat_completion(
            messages=self.messages, stream=True
        ):
            if stop_event is not None and stop_event.is_set():
                break
            if interrupt_check is not None and interrupt_check():
                break
            if self._stop_generation_event.is_set():
                break

            if "error" in response:
                logger.error("Error during streaming: {}", response["error"])
                break

            if "choices" in response and response["choices"]:
                content = response["choices"][0]["delta"].get("content", "")
                if content:
                    assistant_response += content
                    if on_chunk is not None:
                        on_chunk(content)
                    if cli:
                        print(content, end="", flush=True)

        if cli:
            print()

        if assistant_response:
            self.messages.append(
                ChatCompletionRequestAssistantMessage(
                    role="assistant", content=assistant_response
                )
            )

        return assistant_response

    def generate(self, cli: bool = False):
        """Run chat completion synchronously; cancel via :meth:`stop_generation`."""
        self._stop_generation_event.clear()
        self._stream_completion(cli=cli, stop_event=self._stop_generation_event)

    def stop_generation(self):
        self._stop_generation_event.set()
