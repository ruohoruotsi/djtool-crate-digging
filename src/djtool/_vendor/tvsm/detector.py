"""Speech/Music Activity Detector using TVSM CRNN model.

Vendored and adapted from https://github.com/biboamy/TVSM-dataset (Apache 2.0 license).
"""

from __future__ import annotations

import logging

import librosa
import numpy as np
import torch
import torchaudio
import torchvision.transforms as T

from djtool._vendor.tvsm.crnn import CRNN
from djtool._vendor.tvsm.pcen import PCENTransform

logger = logging.getLogger(__name__)

SR = 16000
N_FFT = 1024
HOP_SIZE = 512
CHUNK_DURATION_S = 20


class SMDetector:
    """Speech/Music detector backed by a TVSM CRNN checkpoint.

    Args:
        model_path: Path to the .pt state-dict checkpoint.
        device: Torch device string ("cpu", "cuda", "mps").
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = CRNN()
        logger.info("Loading TVSM model from %s", model_path)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        self.pcen_transform = T.Compose([
            torchaudio.transforms.MelSpectrogram(
                sample_rate=SR, n_fft=N_FFT, hop_length=HOP_SIZE, n_mels=128
            ).to(self.device),
            PCENTransform().to(self.device),
        ])
        logger.info("TVSM detector ready on %s", self.device)

    def predict_audio(self, audio_path: str) -> list[dict]:
        """Run inference on an audio file.

        Args:
            audio_path: Path to the audio file.

        Returns:
            List of dicts with start_time_s, end_time_s, music_prob, speech_prob.
        """
        y, _ = librosa.load(audio_path, sr=SR, mono=True)
        y = np.expand_dims(y, 0)
        audio = torch.from_numpy(y).float().to(self.device)

        audio_pcen = self.pcen_transform(audio)
        chunk_frames = int(SR / HOP_SIZE * CHUNK_DURATION_S)
        n_chunks = int(np.ceil(audio_pcen.shape[-1] / chunk_frames))

        chunks = []
        with torch.inference_mode():
            for i in range(n_chunks):
                chunk = audio_pcen[..., i * chunk_frames : (i + 1) * chunk_frames]
                out = self.model(chunk).detach().cpu()
                chunks.append(out)

        est_label = torch.cat(chunks, -1)
        est_label = torch.sigmoid(est_label)
        est_label = torch.max_pool1d(est_label, 6, 6)

        frame_time = 1 / ((SR / HOP_SIZE) / 6)
        est_np = est_label.detach().cpu().numpy()[0]

        results = []
        for i, frame in enumerate(est_np.T):
            results.append({
                "start_time_s": float(frame_time * i),
                "end_time_s": float(frame_time * (i + 1)),
                "music_prob": round(float(frame[0]), 4),
                "speech_prob": round(float(frame[1]), 4),
            })

        return results
