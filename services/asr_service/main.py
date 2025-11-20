from io import BytesIO
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from fastapi import FastAPI, UploadFile, File
from speechbrain.pretrained import EncoderClassifier


app = FastAPI(title="ASR Service", description="Offline ASR + speaker embeddings prototype")


TARGET_SR = 16_000


def _load_audio_mono_from_bytes(data: bytes, target_sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    """
    Декодирует аудиобайт-стрим в моно-сигнал с заданной частотой дискретизации.
    Используются те же идеи, что и в notebooks/01_asr_and_speaker_id.ipynb.
    """
    with BytesIO(data) as bio:
        audio, sr = sf.read(bio)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    return audio.astype("float32"), sr


def _init_models():
    """
    Инициализация моделей ASR и speaker encoder.
    Выполняется один раз при старте сервиса.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    asr_model = WhisperModel("small", device=device, compute_type=compute_type)
    speaker_encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
    )
    return asr_model, speaker_encoder, device


ASR_MODEL, SPEAKER_ENCODER, DEVICE = _init_models()


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """
    Обрабатывает аудио-чанк встречи:
    - декодирует его до моно 16 кГц;
    - выполняет ASR через faster-whisper;
    - извлекает эмбеддинг говорящего через speechbrain.

    Возвращает:
    - полный текст чанка;
    - простые временные метки (0 и длительность чанка, для детального тайминга лучше использовать info.segments);
    - эмбеддинг говорящего (как список чисел).
    """
    raw = await audio.read()
    waveform, sr = _load_audio_mono_from_bytes(raw, target_sr=TARGET_SR)

    # ASR
    segments, info = ASR_MODEL.transcribe(audio=waveform, language=None)
    segments_list = list(segments)
    text = " ".join(seg.text.strip() for seg in segments_list)

    duration_sec = len(waveform) / float(sr)

    # Speaker embedding
    tensor = torch.from_numpy(waveform).float().unsqueeze(0)
    with torch.no_grad():
        emb = SPEAKER_ENCODER.encode_batch(tensor)
    emb = emb.squeeze(0).squeeze(0).cpu().numpy()

    return {
        "text": text,
        "embedding": emb.tolist(),
        "start": 0.0,
        "end": duration_sec,
    }



