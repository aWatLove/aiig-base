"""
AudioBuffer - класс для управления буферизацией аудио данных.

Отвечает только за хранение и извлечение аудио чанков.
"""

from collections import deque
from typing import List

import numpy as np


class AudioBuffer:
    """
    Буфер для накопления аудио чанков.
    
    Поддерживает:
    - Pre-roll буфер (кольцевой) для сохранения начальных фонем
    - Активный буфер для накопления речи
    """
    
    def __init__(self, pre_roll_size: int = 15):
        """
        Инициализация буфера.
        
        Args:
            pre_roll_size: Размер pre-roll буфера в чанках (~480мс при 15 чанках)
        """
        self.pre_roll_buffer: deque = deque(maxlen=pre_roll_size)
        self.active_buffer: List[np.ndarray] = []
    
    def append_to_pre_roll(self, chunk: np.ndarray) -> None:
        """Добавляет чанк в pre-roll буфер (кольцевой)."""
        self.pre_roll_buffer.append(chunk.copy())
    
    def append_to_active(self, chunk: np.ndarray) -> None:
        """Добавляет чанк в активный буфер."""
        self.active_buffer.append(chunk)
    
    def start_recording(self) -> None:
        """Начинает запись: копирует pre-roll в активный буфер."""
        self.active_buffer = list(self.pre_roll_buffer)
    
    def get_full_audio(self) -> np.ndarray:
        """
        Возвращает полное аудио из активного буфера.
        
        Returns:
            Объединенный массив всех чанков
        """
        if not self.active_buffer:
            return np.array([], dtype=np.float32)
        return np.concatenate(self.active_buffer)
    
    def clear_active(self) -> None:
        """Очищает активный буфер."""
        self.active_buffer = []
    
    def clear_all(self) -> None:
        """Очищает все буферы."""
        self.pre_roll_buffer.clear()
        self.active_buffer = []
    
    def is_empty(self) -> bool:
        """Проверяет, пуст ли активный буфер."""
        return len(self.active_buffer) == 0

