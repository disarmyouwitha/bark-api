import re
import sys
import asyncio
import uvicorn
from typing import Union
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, Optional
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
# bark imports:
from bark.api import generate_audio
from transformers import BertTokenizer
from scipy.io.wavfile import write as write_wav
from bark.generation import SAMPLE_RATE, preload_models, codec_decode, generate_coarse, generate_fine, generate_text_semantic


# load the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")


# Setup FastAPI:
app = FastAPI()
semaphore = asyncio.Semaphore(1)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    message: str


@app.post("/generate")
async def stream_data(req: GenerateRequest):
    while True:
        try:
            # Attempt to acquire the semaphore without waiting
            await asyncio.wait_for(semaphore.acquire(), timeout=0.1)
            break
        except asyncio.TimeoutError:
            print("Server is busy")
            await asyncio.sleep(1)

    try:
        print(req.message)

        # [Enter your prompt and speaker here]:
        #text_prompt = "You're a kid now, you're a squid now!"
        voice_name = "en_speaker_1"

        # simple generation
        audio_array = generate_audio(req.message, history_prompt=voice_name, text_temp=0.7, waveform_temp=0.7)

        # generation with more control
        '''
        x_semantic = generate_text_semantic(
            req.message,
            history_prompt=voice_name,
            temp=0.7,
            top_k=50,
            top_p=0.95,
        )

        x_coarse_gen = generate_coarse(
            x_semantic,
            history_prompt=voice_name,
            temp=0.7,
            top_k=50,
            top_p=0.95,
        )
        x_fine_gen = generate_fine(
            x_coarse_gen,
            history_prompt=voice_name,
            temp=0.5,
        )
        audio_array = codec_decode(x_fine_gen)
        '''

        # save audio
        file_path = "/home/nap/audio.wav"
        write_wav(file_path, SAMPLE_RATE, audio_array)

        # Use the StreamingResponse class to stream the audio file
        file_stream = open(file_path, mode="rb")
        return StreamingResponse(file_stream, media_type="audio/wav")

    except Exception as e:
        return {'response': f"Exception while processing request: {e}"}

    finally:
        semaphore.release()

if __name__ == "__main__":
    # download and load all models
    preload_models(
        text_use_gpu=True,
        text_use_small=False,
        coarse_use_gpu=True,
        coarse_use_small=False,
        fine_use_gpu=True,
        fine_use_small=False,
        codec_use_gpu=True,
        force_reload=False
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7862
    )