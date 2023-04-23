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
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav

# load models once:

# download and load all models
preload_models()

# start fastapi for inference:

# generate audio from text
#text_prompt = """
#     Hello, my name is Suno. And, uh — and I like pizza. [laughs] 
#     But I also have other interests such as playing tic tac toe.
#"""
#audio_array = generate_audio(text_prompt, history_prompt="en_speaker_1")

text_prompt = """
    I have a silky smooth voice, and today I will tell you about 
    the exercise regimen of the common sloth.
"""
audio_array = generate_audio(text_prompt, history_prompt="en_speaker_1")

# return audio as audio stream:

write_wav("/home/nap/audio.wav", SAMPLE_RATE, audio_array)