"""
Реализация ASR движка на базе Faster-Whisper.

Только транскрибация - без метрик и побочных эффектов.
"""

from typing import List, Union

import numpy as np
from faster_whisper import WhisperModel

from v2t.core.contract import IASREngine
from v2t.core.dto import Utterance


class FasterWhisperEngine(IASREngine):
    """
    Реализация ASR движка на базе Faster-Whisper.

    Отвечает только за преобразование аудио в текст.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 4
    ):
        """
        Инициализация Faster-Whisper модели.

        Args:
            model_size: Размер модели (tiny, base, small, medium, large)
            device: Устройство для вычислений (cpu, cuda)
            compute_type: Тип вычислений (int8, float16, float32)
            cpu_threads: Количество потоков CPU (оптимально 4 для M4)
        """
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads
        )

    def transcribe(
        self, 
        audio: Union[np.ndarray, str],
        beam_size: int = 1,
        vad_filter: bool = False
    ) -> List[Utterance]:
        """
        Распознает речь в аудио данных.

        Args:
            audio: Аудио данные (numpy array или путь к файлу)
            beam_size: Ширина луча для поиска (1, 2, 5 - больше = точнее, но медленнее)
            vad_filter: Использовать встроенный VAD фильтр (True/False)

        Returns:
            Список распознанных фраз (Utterance)
        """
        segments, _ = self.model.transcribe(
            audio,
            beam_size=beam_size,
            vad_filter=vad_filter
        )

        return [
            Utterance(
                text=segment.text.strip(),
                start_time=round(segment.start, 2),
                end_time=round(segment.end, 2),
                confidence=round(np.exp(segment.avg_logprob), 2)
            )
            for segment in segments
        ]
