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
from time import time
from tqdm.auto import tqdm
#from IPython.display import Audio
from scipy.io.wavfile import write as write_wav
import os
import numpy as np

#Split text it into chunks of a desired length trying to keep sentences intact.
def split_and_recombine_text(text, desired_length=100, max_length=150):
    # from https://github.com/neonbjb/tortoise-tts
    
    # normalize text, remove redundant whitespace and convert non-ascii quotes to ascii
    text = re.sub(r"\n\n+", "\n", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[“”]", '"', text)

    rv = []
    in_quote = False
    current = ""
    split_pos = []
    pos = -1
    end_pos = len(text) - 1

    def seek(delta):
        nonlocal pos, in_quote, current
        is_neg = delta < 0
        for _ in range(abs(delta)):
            if is_neg:
                pos -= 1
                current = current[:-1]
            else:
                pos += 1
                current += text[pos]
            if text[pos] == '"':
                in_quote = not in_quote
        return text[pos]

    def peek(delta):
        p = pos + delta
        return text[p] if p < end_pos and p >= 0 else ""

    def commit():
        nonlocal rv, current, split_pos
        rv.append(current)
        current = ""
        split_pos = []

    while pos < end_pos:
        c = seek(1)
        # do we need to force a split?
        if len(current) >= max_length:
            if len(split_pos) > 0 and len(current) > (desired_length / 2):
                # we have at least one sentence and we are over half the desired length, seek back to the last split
                d = pos - split_pos[-1]
                seek(-d)
            else:
                # no full sentences, seek back until we are not in the middle of a word and split there
                while c not in "!?.\n " and pos > 0 and len(current) > desired_length:
                    c = seek(-1)
            commit()
        # check for sentence boundaries
        elif not in_quote and (c in "!?\n" or (c == "." and peek(1) in "\n ")):
            # seek forward if we have consecutive boundary markers but still within the max length
            while (
                pos < len(text) - 1 and len(current) < max_length and peek(1) in "!?."
            ):
                c = seek(1)
            split_pos.append(pos)
            if len(current) >= desired_length:
                commit()
        # treat end of quote as a boundary if its followed by a space or newline
        elif in_quote and peek(1) == '"' and peek(2) in "\n ":
            seek(2)
            split_pos.append(pos)
    rv.append(current)

    # clean up, remove lines with only whitespace or punctuation
    rv = [s.strip() for s in rv]
    rv = [s for s in rv if len(s) > 0 and not re.match(r"^[\s\.,;:!?]*$", s)]

    return rv


# generation with more control:
def generate_with_settings(text_prompt, semantic_temp=0.7, semantic_top_k=50, semantic_top_p=0.95, coarse_temp=0.7, coarse_top_k=50, coarse_top_p=0.95, fine_temp=0.5, voice_name=None, use_semantic_history_prompt=True, use_coarse_history_prompt=True, use_fine_history_prompt=True, output_full=False):
    x_semantic = generate_text_semantic(
        text_prompt,
        history_prompt=voice_name if use_semantic_history_prompt else None,
        temp=semantic_temp,
        top_k=semantic_top_k,
        top_p=semantic_top_p,
    )

    x_coarse_gen = generate_coarse(
        x_semantic,
        history_prompt=voice_name if use_coarse_history_prompt else None,
        temp=coarse_temp,
        top_k=coarse_top_k,
        top_p=coarse_top_p,
    )

    x_fine_gen = generate_fine(
        x_coarse_gen,
        history_prompt=voice_name if use_fine_history_prompt else None,
        temp=fine_temp,
    )

    if output_full:
        full_generation = {
            'semantic_prompt': x_semantic,
            'coarse_prompt': x_coarse_gen,
            'fine_prompt': x_fine_gen,
        }
        return full_generation, codec_decode(x_fine_gen)

    return codec_decode(x_fine_gen)


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


@app.get("/check")
async def stream_data():
    return { "tts": "ok" }

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
        # [Enter your prompt and speaker here]:
        print(req.message)
        #text_prompt = "You're a kid now, you're a squid now!"
        voice_name = "en_speaker_1"

        # simple generation
        audio_array = generate_audio(req.message, history_prompt=voice_name, text_temp=0.7, waveform_temp=0.7)

        # save audio
        file_path = "audio.wav"
        # ^ start using random file name, put in output/<hash>.wav
        write_wav(file_path, SAMPLE_RATE, audio_array)

        # Use the StreamingResponse class to stream the audio file
        file_stream = open(file_path, mode="rb")
        return StreamingResponse(file_stream, media_type="audio/wav")

    except Exception as e:
        return {'response': f"Exception while processing request: {e}"}

    finally:
        semaphore.release()

# need to stream each chunk.. probably unessisary to combine chunks back together and save unless we want to preserve history.

@app.post("/agenerate")
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
        req.message = """The Uncharted Land of Discovery: A Journey Through Time and Space
[clears throat]
Chapter 1: The Dawn of Curiosity
[takes breath]
Since the dawn of humankind, our species has been driven by a powerful force: curiosity. It is an innate, unquenchable desire to explore, understand, and unravel the mysteries of the world around us. This primal urge has led us on countless adventures, pushing us to the farthest reaches of our planet and beyond.

Early humans, huddled around a flickering fire, gazed up at the night sky and wondered what those twinkling lights were. They had no idea that their curiosity would eventually propel us into the vast, uncharted realm of space. As time progressed, our ancestors began to explore their surroundings, venturing beyond their caves and settlements, driven by the need to discover what lay beyond the horizon.

hapter 2: The Age of Exploration"""
        print(req.message)

        # generation settings
        voice_name = 'speaker_4'
        #---
        semantic_temp = 0.7
        semantic_top_k = 50
        semantic_top_p = 0.95
        #---
        coarse_temp = 0.7
        coarse_top_k = 50
        coarse_top_p = 0.95
        #---
        fine_temp = 0.5
        #---
        use_semantic_history_prompt = True
        use_coarse_history_prompt = True
        use_fine_history_prompt = True
        use_last_generation_as_history = True
        #-------

        # split req.message into lines:
        texts = split_and_recombine_text(req.message)

        cnt = 0
        all_parts = []
        for i, text in tqdm(enumerate(texts), total=len(texts)):
            cnt = cnt+1
            print("CNT: {0}".format(cnt), flush=True)
            full_generation, audio_array = generate_with_settings(
                text,
                semantic_temp=semantic_temp,
                semantic_top_k=semantic_top_k,
                semantic_top_p=semantic_top_p,
                coarse_temp=coarse_temp,
                coarse_top_k=coarse_top_k,
                coarse_top_p=coarse_top_p,
                fine_temp=fine_temp,
                voice_name=voice_name,
                use_semantic_history_prompt=use_semantic_history_prompt,
                use_coarse_history_prompt=use_coarse_history_prompt,
                use_fine_history_prompt=use_fine_history_prompt,
                output_full=True
            )
            print("after get", flush=True)
            if use_last_generation_as_history:
                # save to npz
                os.makedirs('_temp', exist_ok=True)
                np.savez_compressed(
                    '_temp/history.npz',
                    semantic_prompt=full_generation['semantic_prompt'],
                    coarse_prompt=full_generation['coarse_prompt'],
                    fine_prompt=full_generation['fine_prompt'],
                )
                voice_name = '_temp/history.npz'
            all_parts.append(audio_array)
            print("after history", flush=True)

            # instead of waiting until the end we save the file so that we can start streaming this part.
            fp = "/home/nap/audio{1}.wav".format(cnt)
            print("writing file: {0}".format(fp), flush=True)
            write_wav(fp, SAMPLE_RATE, audio_array)
            print("file saved!", flush=True)
            file_stream = open(fp, mode="rb")
            return StreamingResponse(file_stream, media_type="audio/wav")

        #audio_array = np.concatenate(all_parts, axis=-1)

        # save audio
        #write_wav(out_filepath, SAMPLE_RATE, audio_array)


        # Use the StreamingResponse class to stream the audio file
        #file_stream = open(file_path, mode="rb")
        #return StreamingResponse(file_stream, media_type="audio/wav")

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