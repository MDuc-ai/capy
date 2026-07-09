import asyncio

from capy.core.asr import CapyASR, SAMPLE_RATE
from capy.core.chat_engine import CapyChatEngine
from capy.core.llm import CapyLLM
from capy.core.microphone import MicrophoneInput


async def test_chat_engine():
    llm = CapyLLM()
    llm.load_llm()

    asr = CapyASR()
    microphone = MicrophoneInput.from_default_device(sampling_rate=SAMPLE_RATE)

    chat_engine = CapyChatEngine(microphone=microphone, asr=asr, llm=llm)
    try:
        await chat_engine.run_interactive_cli()
    finally:
        llm.offload_llm()
        microphone.close()


if __name__ == "__main__":
    asyncio.run(test_chat_engine())
