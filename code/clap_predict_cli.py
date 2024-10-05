"""
Quick experiments to evaluate the CLAP & SMAD classifications
"""
import argparse
import glob
import os.path

import librosa
import numpy as np
from transformers import AutoProcessor, ClapModel

np.float_ = np.float64
from pathlib import Path

model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")


def clap_inference(input_text, audio_sample, sr):
    inputs = processor(text=input_text, audios=audio_sample, return_tensors="pt", padding=True, sampling_rate=sr)
    outputs = model(**inputs)
    logits_per_audio = outputs.logits_per_audio  # this is the audio-text similarity score
    probs = logits_per_audio.softmax(dim=-1)  # we can take the softmax to get the label probabilities
    return probs.cpu().detach().numpy().squeeze()


djtool_description_list = [
    "acapella, expressively sung human vocal with background instrumental music tracks",
    # "purely acoustic sung vocals, vocal-only performance",
    "piano, synths or strings, instrumental, guitar",
    "drums, a drum loop, drum solo, breakbeat, percussive elements",
    "beatboxing",
    "siren, riser sound effects, whoosh, crash, synthetic, transitional effect",
    "vinyl scratch loop, turnatablist DJ battle sounds",
    "a high energy, high tension, climactic, massive EDM drop"
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="input audio file dir")
    parser.add_argument(dest='input_segments_dir')
    args = parser.parse_args()

    wav_pattern = str(Path(__file__).parent.joinpath(args.input_segments_dir)) + "/*.wav"
    seg_file_list = glob.glob(wav_pattern, recursive=False)
    seg_file_list.sort()
    for seg_file in seg_file_list:
        audio_segment, sr = librosa.load(seg_file, sr=48000, mono=True)
        # optionally scale the duration by exactly {23, 18, 13, 8, 3} seconds
        # audio_segment = audio_segment[:sr*23]

        duration = len(audio_segment) / sr
        print("duration {}".format(duration))
        if duration < 3:
            print("{} is too short\n".format(seg_file))
            continue

        # iterate over sufficiently long segments
        probs = clap_inference(input_text=djtool_description_list, audio_sample=audio_segment, sr=sr)
        print("{}:      {}\n".format(os.path.basename(seg_file), np.round(probs, 2)))
