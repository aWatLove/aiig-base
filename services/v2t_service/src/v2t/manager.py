"""
StreamManager - оркестратор для управления потоком аудио данных.

Только оркестрация: реагирует на события, управляет состояниями, вызывает ASR.
Без метрик и сложной логики буферизации.
"""

import asyncio
from enum import Enum
from typing import Optional, List, Callable

from v2t.core.contract import IASREngine, IVADFilter
from v2t.core.dto import AudioChunk, Utterance
from v2t.core.buffer import AudioBuffer


class StreamState(Enum):
    """Состояния машины состояний StreamManager."""
    IDLE = "idle"
    RECORDING = "recording"


class StreamManager:
    """
    Оркестратор для управления потоком аудио данных.
    
    Отвечает только за:
    - Реакцию на события (приход чанка, обнаружение речи)
    - Управление состояниями (IDLE -> RECORDING)
    - Вызов ASR при финализации
    """
    
    SILENCE_THRESHOLD_CHUNKS = 20
    
    def __init__(
        self,
        asr: IASREngine,
        vad: Optional[IVADFilter] = None,
        on_utterance: Optional[Callable[[List[Utterance]], None]] = None,
        pre_roll_size: int = 15
    ):
        """
        Инициализация StreamManager.
        
        Args:
            asr: Движок распознавания речи (ASR)
            vad: Опциональный фильтр VAD для определения речи
            on_utterance: Callback для обработки результатов распознавания
            pre_roll_size: Размер pre-roll буфера в чанках
        """
        self.asr = asr
        self.vad = vad
        self.on_utterance = on_utterance
        
        self.buffer = AudioBuffer(pre_roll_size=pre_roll_size)
        self.state = StreamState.IDLE
        self.silence_chunks_count = 0

    async def consume_chunk(self, chunk: AudioChunk) -> None:
        """
        Обрабатывает входящий аудио-чанк.
        
        Args:
            chunk: Аудио чанк для обработки
        """
        self.buffer.append_to_pre_roll(chunk.data)
        
        if self.vad is None:
            self.buffer.append_to_active(chunk.data)
            return
        
        has_speech = self.vad.is_speech(chunk.data)
        
        if self.state == StreamState.IDLE:
            if has_speech:
                await self._start_recording()
        
        elif self.state == StreamState.RECORDING:
            if has_speech:
                self.buffer.append_to_active(chunk.data)
                self.silence_chunks_count = 0
            else:
                self.silence_chunks_count += 1
                self.buffer.append_to_active(chunk.data)
                
                if self.silence_chunks_count >= self.SILENCE_THRESHOLD_CHUNKS:
                    await self._finalize_utterance()

    async def _start_recording(self) -> None:
        """Начало записи: копируем pre-roll в активный буфер."""
        self.state = StreamState.RECORDING
        self.silence_chunks_count = 0
        self.buffer.start_recording()
        
        if self.vad:
            self.vad.reset()

    async def _finalize_utterance(self) -> None:
        """Финализация фразы: отправка в ASR и вызов callback."""
        if self.buffer.is_empty():
            self.state = StreamState.IDLE
            return
        
        full_audio = self.buffer.get_full_audio()
        results = await asyncio.to_thread(self.asr.transcribe, full_audio)
        
        if self.on_utterance:
            self.on_utterance(results)
        
        self.buffer.clear_active()
        self.state = StreamState.IDLE
        self.silence_chunks_count = 0

    def reset(self) -> None:
        """Сброс всех буферов и состояний."""
        self.buffer.clear_all()
        self.state = StreamState.IDLE
        self.silence_chunks_count = 0
        if self.vad:
            self.vad.reset()
