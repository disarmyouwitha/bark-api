
from bark.generation import SAMPLE_RATE, preload_models, codec_decode, generate_coarse, generate_fine, generate_text_semantic

import re
def split_and_recombine_text(text, desired_length=100, max_length=150):
    # from https://github.com/neonbjb/tortoise-tts
    """Split text it into chunks of a desired length trying to keep sentences intact."""
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

def generate_with_settings(text_prompt, semantic_temp=0.7, semantic_top_k=50, semantic_top_p=0.95, coarse_temp=0.7, coarse_top_k=50, coarse_top_p=0.95, fine_temp=0.5, voice_name=None, use_semantic_history_prompt=True, use_coarse_history_prompt=True, use_fine_history_prompt=True, output_full=False):
    # generation with more control
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

# `[laughter]`
# - `[laughs]`
# - `[sighs]`
# - `[music]`
# - `[gasps]`
# - `[clears throat]`
# - `—` or `...` for hesitations
# - `♪` for song lyrics

text = """The Uncharted Land of Discovery: A Journey Through Time and Space
[clears throat]
Chapter 1: The Dawn of Curiosity
[takes breath]
Since the dawn of humankind, our species has been driven by a powerful force: curiosity. It is an innate, unquenchable desire to explore, understand, and unravel the mysteries of the world around us. This primal urge has led us on countless adventures, pushing us to the farthest reaches of our planet and beyond.

Early humans, huddled around a flickering fire, gazed up at the night sky and wondered what those twinkling lights were. They had no idea that their curiosity would eventually propel us into the vast, uncharted realm of space. As time progressed, our ancestors began to explore their surroundings, venturing beyond their caves and settlements, driven by the need to discover what lay beyond the horizon.

hapter 2: The Age of Exploration

The Age of Exploration marked a turning point in human history, as brave souls took to the seas in search of new lands, wealth, and knowledge. Pioneers like Christopher Columbus, Vasco da Gama, and Ferdinand Magellan set sail on perilous voyages, pushing the boundaries of what was known and understood.
[clears throat]
These intrepid explorers discovered new continents, mapped out previously unknown territories, and encountered diverse cultures. They also established trade routes, allowing for the exchange of goods, ideas, and innovations between distant societies. The Age of Exploration was not without its dark moments, however, as conquest, colonization, and exploitation often went hand in hand with discovery.
[clears throat]
Chapter 3: The Scientific Revolution
[laughs]
The Scientific Revolution was a period of profound change, as humanity began to question long-held beliefs and seek empirical evidence. Pioneers like Galileo Galilei, Isaac Newton, and Johannes Kepler sought to understand the natural world through observation, experimentation, and reason.
[sighs]
Their discoveries laid the foundation for modern science, transforming the way we view the universe and our place within it. New technologies, such as the telescope and the microscope, allowed us to peer deeper into the cosmos and the microscopic world, further expanding our understanding of reality.
[gasps]
Chapter 4: The Information Age

The Information Age, sometimes referred to as the Digital Age, has revolutionized the way we communicate, learn, and access knowledge. With the advent of the internet and personal computers, information that was once reserved for the privileged few is now available to the masses.
...
This democratization of knowledge has led to an explosion of innovation, as ideas and information are shared across borders and cultures at lightning speed. The Information Age has also brought new challenges, as the rapid pace of technological advancements threatens to outpace our ability to adapt and raises questions about the ethical implications of our increasingly interconnected world.
[laughter]
Chapter 5: The Final Frontier
[clears throat]
As our knowledge of the universe expands, so too does our desire to explore the cosmos. Space exploration has come a long way since the first successful satellite, Sputnik, was launched in 1957. We have landed humans on the moon, sent probes to the far reaches of our solar system, and even glimpsed distant galaxies through powerful telescopes.

The future of space exploration is filled with possibilities, from establishing colonies on Mars to the search for extraterrestrial life. As we venture further into the unknown, we continue to be driven by the same curiosity that has propelled us throughout history, always seeking to uncover the secrets of the universe and our place within it.
...
In conclusion, the human journey is one of discovery, driven by our innate curiosity and desire to understand the world around us. From the dawn of our species to the present day, we have continued to explore, learn, and adapt, pushing the boundaries of what is known and possible. As we continue to unravel the mysteries of the cosmos, our spirit."""

# Chunk the text into smaller pieces then combine the generated audio
from time import time
from tqdm.auto import tqdm
#from IPython.display import Audio
from scipy.io.wavfile import write as write_wav
import os
import numpy as np

# generation settings
voice_name = 'speaker_4'
out_filepath = 'audio/audio.wav'

semantic_temp = 0.7
semantic_top_k = 50
semantic_top_p = 0.95

coarse_temp = 0.7
coarse_top_k = 50
coarse_top_p = 0.95

fine_temp = 0.5

use_semantic_history_prompt = True
use_coarse_history_prompt = True
use_fine_history_prompt = True

use_last_generation_as_history = True

texts = split_and_recombine_text(text)

all_parts = []
for i, text in tqdm(enumerate(texts), total=len(texts)):
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

audio_array = np.concatenate(all_parts, axis=-1)

# save audio
write_wav(out_filepath, SAMPLE_RATE, audio_array)

# play audio
#Audio(audio_array, rate=SAMPLE_RATE)