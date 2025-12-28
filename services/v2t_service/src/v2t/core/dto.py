from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Utterance:
    """
    Результат распознавания одной фразы.
    
    Attributes:
        text: Распознанный текст
        start_time: Время начала фразы в секундах
        end_time: Время окончания фразы в секундах
        confidence: Уверенность распознавания (0.0 - 1.0)
        speaker_id: Опциональный идентификатор спикера
    """
    text: str
    start_time: float
    end_time: float
    confidence: float
    speaker_id: Optional[str] = None


@dataclass
class AudioChunk:
    """
    Кусочек аудио данных для потоковой обработки.
    
    Attributes:
        data: Аудио данные (numpy array, float32, 16kHz)
        timestamp: Временная метка чанка в секундах
    """
    data: np.ndarray  # float32, 16kHz, размер 512 сэмплов (32мс)
    timestamp: float

