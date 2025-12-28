"""
Реализация VAD фильтра на базе Silero VAD.

Только определение речи - без ленивой загрузки и скрытых состояний.
"""

import numpy as np

from v2t.core.contract import IVADFilter


class SileroGatekeeper(IVADFilter):
    """
    Реализация Gatekeeper на базе Silero VAD.
    
    Определяет наличие речи в аудио чанке.
    Модель должна быть загружена и передана при инициализации.
    """
    
    SAMPLE_RATE = 16000
    
    def __init__(self, model, threshold: float = 0.5):
        """
        Инициализация Silero VAD фильтра.
        
        Args:
            model: Загруженная модель Silero VAD (готовый объект)
            threshold: Порог вероятности для определения речи (0.0 - 1.0)
        """
        self.model = model
        self.threshold = threshold

    def is_speech(self, chunk: np.ndarray) -> bool:
        """
        Определяет, содержит ли чанк речь.
        
        Args:
            chunk: Аудио данные (numpy array, float32, 16kHz)
            
        Returns:
            True если обнаружена речь, False иначе
        """
        speech_prob = self.model(chunk, self.SAMPLE_RATE)
        return speech_prob > self.threshold
    
    def reset(self) -> None:
        """
        Сбрасывает внутреннее состояние фильтра.
        
        Если модель имеет внутреннее состояние (например, RNN),
        его нужно сбросить здесь.
        """
        if hasattr(self.model, 'reset_states'):
            self.model.reset_states()
