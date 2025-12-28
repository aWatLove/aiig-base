"""
Voice-to-Text (V2T) Service

Сервис для преобразования речи в текст с поддержкой:
- ASR (Automatic Speech Recognition) через Faster-Whisper
- VAD (Voice Activity Detection) через Silero
- Управление потоком аудио данных через StreamManager
"""

from v2t.core.contract import IASREngine, IVADFilter
from v2t.core.dto import AudioChunk, Utterance
from v2t.core.buffer import AudioBuffer
from v2t.manager import StreamManager
from v2t.asr.whisper_impl import FasterWhisperEngine
from v2t.vad.silero_impl import SileroGatekeeper

__all__ = [
    'IASREngine',
    'IVADFilter',
    'AudioChunk',
    'Utterance',
    'AudioBuffer',
    'StreamManager',
    'FasterWhisperEngine',
    'SileroGatekeeper',
]

