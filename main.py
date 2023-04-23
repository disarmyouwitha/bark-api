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

# simple generation
#audio_array = generate_audio(text_prompt, history_prompt=voice_name, text_temp=0.7, waveform_temp=0.7)

# generation with more control
x_semantic = generate_text_semantic(
    text_prompt,
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


# save audio
filepath = "/home/nap/audio.wav"
write_wav(filepath, SAMPLE_RATE, audio_array)