"""
Quick experiments to evaluate the CLAP & SMAD classifications
"""
import argparse
import os.path
import sys
import librosa
import soundfile as sf
from transformers import AutoProcessor, ClapModel
import numpy as np
np.float_ = np.float64

import msaf

model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")


# input_text = ["Sound of a dog", "Sound of vaccum cleaner"]
def clap_inference(input_text, audio_sample, sr):
    inputs = processor(text=input_text, audios=audio_sample, return_tensors="pt", padding=True, sampling_rate=sr)
    outputs = model(**inputs)
    logits_per_audio = outputs.logits_per_audio  # this is the audio-text similarity score
    probs = logits_per_audio.softmax(dim=-1)  # we can take the softmax to get the label probabilities
    print(probs)
    return probs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="input audio file path")
    parser.add_argument(dest='input_audio_path')
    args = parser.parse_args()
    sig, sr = librosa.load(args.input_audio_path, sr=48000, mono=True)
    print("sample rate: {}".format(sr))

    clap_inference(input_text=["a drum beat", "acapella or expressively sung vocal tracks", "an instrumental passage with tonal instruments"],
                   audio_sample=sig,
                   sr=sr)

    # Segment the file using the default MSAF parameters
    boundaries, labels = msaf.process(sig)
    print(boundaries)