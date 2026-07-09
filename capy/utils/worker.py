from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from loguru import logger


def _noop_interrupt() -> None:
    """Default barge-in hook: does nothing until an orchestrator injects a real one."""


class InterruptibleEvent:
    """Queue item that can be marked interrupted before or during processing."""

    def __init__(self, data: Any = None, is_interruptible: bool = True) -> None:
        self.data = data
        self.is_interruptible = is_interruptible
        self._interrupted = False

    def interrupt(self) -> None:
        self._interrupted = True

    def is_interrupted(self) -> bool:
        return self._interrupted


class AbstractWorker(ABC):
    """
    A generic processor - knows only how to consume typed items.
    In order for a worker to process items, clients must invoke start() and tear down with terminate()
    """

    @abstractmethod
    def start(self):
        raise NotImplementedError

    @abstractmethod
    def consume_nonblocking(self, item: Any):
        raise NotImplementedError

    async def terminate(self):
        pass


class QueueConsumer(AbstractWorker):
    def __init__(
        self,
        input_queue: Optional[asyncio.Queue[Any]] = None,
    ) -> None:
        self.input_queue: asyncio.Queue[Any] = input_queue or asyncio.Queue()

    def consume_nonblocking(self, item: Any):
        self.input_queue.put_nowait(item)

    def start(self):
        pass


class AsyncWorker(AbstractWorker):
    """
    Async processor with an ``input_queue`` it reads from and an ``output_queue`` it writes to.

    Components are chained by sharing queues: assign the ``output_queue`` of one worker as
    the ``input_queue`` of the next (see :func:`link`). Subclasses publish results onto
    ``output_queue`` so they become the downstream worker's input.
    """

    def __init__(
        self,
    ) -> None:
        self.worker_task: Optional[asyncio.Task] = None
        self.input_queue: asyncio.Queue[Any] = asyncio.Queue()
        self.output_queue: asyncio.Queue[Any] = asyncio.Queue()
        # Reference to the pipeline-wide barge-in. An orchestrator injects its own
        # ``broadcast_interrupt`` here so this worker can interrupt the others when needed.
        self.broadcast_interrupt: Callable[[], None] = _noop_interrupt

    def start(self) -> asyncio.Task:
        self.worker_task = asyncio.create_task(
            self._run_loop(),
        )
        if not self.worker_task:
            raise Exception("Worker task not created")
        return self.worker_task

    def consume_nonblocking(self, item: Any):
        self.input_queue.put_nowait(item)

    async def _run_loop(self):
        raise NotImplementedError

    async def terminate(self):
        if self.worker_task:
            return self.worker_task.cancel()

        return False


class InterruptibleWorker(AsyncWorker):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.current_task: Optional[asyncio.Task] = None
        self.interruptible_event: Optional[InterruptibleEvent] = None

    async def _run_loop(self):
        while True:
            try:
                item = await self.input_queue.get()
            except asyncio.CancelledError:
                return

            # Upstream workers publish raw data onto their output_queue; wrap it so the
            # interruptible machinery has a uniform item type.
            if not isinstance(item, InterruptibleEvent):
                item = InterruptibleEvent(data=item)

            if item.is_interrupted():
                continue

            self.interruptible_event = item
            self.current_task = asyncio.create_task(self.process(item))

            try:
                await self.current_task
            except asyncio.CancelledError:
                # A barge-in cancels only the in-flight response (the sub-task ends up
                # cancelled); keep serving the next item. If instead the worker task
                # itself is being terminated, the sub-task is still running, so cancel
                # it and exit the loop.
                if self.current_task.cancelled():
                    self.current_task = None
                    continue
                self.current_task.cancel()
                self.current_task = None
                return
            except Exception:
                logger.exception("InterruptibleWorker", exc_info=True)

            self.current_task = None

    async def process(self, item: Any):
        """
        Publish results onto output queue.
        Calls to async function / task should be able to handle asyncio.CancelledError gracefully:
        """
        raise NotImplementedError

    def cancel_current_task(self):
        """Cancel the in-flight response iff it is currently processing an interruptible item.

        Returns True if a running task was cancelled, False otherwise. Safe to call when
        nothing is being processed.
        """
        if (
            self.current_task
            and not self.current_task.done()
            and self.interruptible_event is not None
            and self.interruptible_event.is_interruptible
        ):
            return self.current_task.cancel()

        return False


InterruptibleAsyncWorker = InterruptibleWorker


def link(*components: Any) -> None:
    """
    Chain components by sharing queues.

    For each adjacent pair, the upstream component's ``output_queue`` becomes the
    downstream component's ``input_queue``, so items produced by one are read directly
    by the next with no relay task in between::

        link(microphone, asr, llm)
        # asr.input_queue is microphone.output_queue
        # llm.input_queue is asr.output_queue
    """
    for upstream, downstream in zip(components, components[1:]):
        downstream.input_queue = upstream.output_queue
