from dataclasses import dataclass, field
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="Orchestrator Service", description="Manages meeting context and triggers image generation")


class UtteranceIn(BaseModel):
    text: str
    speaker_id: str
    timestamp: float


@dataclass
class Utterance:
    text: str
    speaker_id: str
    timestamp: float


@dataclass
class MeetingState:
    utterances: List[Utterance] = field(default_factory=list)
    max_history_seconds: float = 300.0

    def add_utterance(self, utt: Utterance):
        self.utterances.append(utt)
        self._trim_history()

    def _trim_history(self):
        """
        Обрезает историю по времени так, чтобы хранились только последние max_history_seconds.
        Предполагается, что timestamp растёт во времени.
        """
        if not self.utterances:
            return
        latest_ts = self.utterances[-1].timestamp
        cutoff = latest_ts - self.max_history_seconds
        self.utterances = [u for u in self.utterances if u.timestamp >= cutoff]

    def build_prompt(self) -> str:
        parts = [f"[{u.speaker_id}] {u.text}" for u in self.utterances[-10:]]
        conversation = " \n".join(parts)
        prompt = (
            "An illustration summarizing the ongoing conversation between participants. "
            "Focus on the main themes and emotions. Conversation: " + conversation
        )
        return prompt


MEETING_STATE = MeetingState()


@app.post("/ingest_utterance")
async def ingest_utterance(utt: UtteranceIn):
    """
    Принимает новую реплику от ASR-сервиса и обновляет состояние встречи.
    """
    MEETING_STATE.add_utterance(
        Utterance(text=utt.text, speaker_id=utt.speaker_id, timestamp=utt.timestamp)
    )
    return {"status": "ok", "num_utterances": len(MEETING_STATE.utterances)}


@app.post("/trigger_generation")
async def trigger_generation():
    """
    Точка вызова генерации:
    - в дальнейшем будет ходить в image_service;
    - здесь строится промпт по текущему состоянию встречи.
    """
    prompt = MEETING_STATE.build_prompt()
    # TODO: сделать запрос в image_service и вернуть реальный результат
    return {"prompt": prompt, "note": "image generation call is stubbed"}



