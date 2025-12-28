from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from v2t.core.dto import Utterance


class IASREngine(ABC):
    """Интерфейс для движка автоматического распознавания речи (ASR)."""
    
    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> List['Utterance']:
        """
        Распознает речь в аудио данных.
        
        Args:
            audio: Аудио данные в формате numpy array (float32, 16kHz)
            
        Returns:
            Список распознанных фраз (Utterance)
        """
        pass


class IVADFilter(ABC):
    """Интерфейс для фильтра определения активности речи (VAD)."""
    
    @abstractmethod
    def is_speech(self, chunk: np.ndarray) -> bool:
        """
        Определяет, содержит ли чанк речь.
        
        Args:
            chunk: Аудио чанк (numpy array, float32, 16kHz)
            
        Returns:
            True если обнаружена речь, False иначе
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Сбрасывает внутреннее состояние фильтра.
        Используется при переходе между состояниями в StreamManager.
        """
        pass

