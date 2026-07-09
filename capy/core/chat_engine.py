from __future__ import annotations

import asyncio

from capy.core.asr import CapyASR
from capy.core.cli_output import CapyCliOutput
from capy.core.llm import CapyLLM
from capy.core.microphone import MicrophoneInput
from capy.utils.worker import AsyncWorker, InterruptibleWorker, link


class CapyChatEngine:
    """
    Async orchestrator for Capy's voice assistant.

    The component order is fixed and wired by sharing queues between neighbours::

        MicrophoneInput --output_queue--> CapyASR --output_queue--> CapyLLM

    Data flows without any relay task: float32 PCM goes mic -> asr, transcript strings
    go asr -> llm, and the assistant's response chunks are published on ``llm.output_queue``.

    The CLI output worker sits outside this chain; ASR and LLM publish to it for display.
    """

    def __init__(
        self,
        microphone: MicrophoneInput,
        asr: CapyASR,
        llm: CapyLLM,
        cli_output: CapyCliOutput | None = None,
    ):
        self.microphone = microphone
        self.asr = asr
        self.llm = llm
        self.cli_output = cli_output or CapyCliOutput()

        link(microphone, asr, llm)

        self.asr.set_cli_output(self.cli_output)
        self.llm.set_cli_output(self.cli_output)

        self.workers = [asr, llm]

        for worker in self.workers:
            if isinstance(worker, AsyncWorker):
                worker.broadcast_interrupt = self.broadcast_interrupt

    def broadcast_interrupt(self):
        """Interrupt every worker currently processing an interruptible item (barge-in)."""
        for worker in self.workers:
            if isinstance(worker, InterruptibleWorker):
                worker.cancel_current_task()

    async def start(self):
        """Start the ASR and LLM workers (the microphone streams as soon as it is created)."""
        self.asr.start()
        self.llm.start()
        self.cli_output.start()

    async def stop(self):
        await self.cli_output.terminate()
        await self.llm.terminate()
        await self.asr.terminate()

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *_args):
        await self.stop()

    def interrupt_generation(self):
        """Cancel the in-flight response across the pipeline (barge-in)."""
        self.broadcast_interrupt()

    async def run_interactive_cli(self):
        """
        Run an interactive session.

        Voice input flows automatically through the shared-queue pipeline. Typed lines are
        injected into the LLM as well. Type 'quit', 'exit', or 'bye' (or Ctrl+C) to leave.
        """
        if not self.llm.is_loaded:
            raise ValueError("No model loaded!")

        await self.start()
        print(
            "\n".join(
                [
                    "Interactive CLI:",
                    "- Speak into the microphone; transcripts are answered automatically.",
                    "- Or type a message and press Enter to send it.",
                    "- Type 'quit', 'exit', or 'bye' (or press Ctrl+C) to exit.",
                ]
            )
        )

        try:
            while True:
                text = (await asyncio.to_thread(input)).strip()
                if text.lower() in ("quit", "exit", "bye"):
                    break
                if text:
                    self.broadcast_interrupt()
                    self.llm.consume_transcription(text)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
        finally:
            await self.stop()
