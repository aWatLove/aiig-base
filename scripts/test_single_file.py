"""
Скрипт для тестирования ASR на одном аудио файле (1 итерация).

Позволяет прослушать файл и сравнить с результатом распознавания.
"""

import sys
from pathlib import Path
from typing import List

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "v2t_service" / "src"))

from v2t.asr.whisper_impl import FasterWhisperEngine
from v2t.core.dto import Utterance


def load_audio(path: Path, target_sr: int = 16000) -> np.ndarray:
    """
    Загружает аудио файл и приводит к нужному формату.
    
    Args:
        path: Путь к аудио файлу
        target_sr: Целевая частота дискретизации
        
    Returns:
        Аудио данные (float32, mono, target_sr)
    """
    # Пробуем разные библиотеки для загрузки
    try:
        audio, sr = sf.read(str(path))
    except:
        # Если soundfile не работает, используем librosa
        audio, sr = librosa.load(str(path), sr=None, mono=False)
    
    # Приводим к моно
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    
    # Приводим к нужной частоте дискретизации
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    # Нормализация
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    return audio.astype(np.float32)


def format_utterances(utterances: List[Utterance]) -> str:
    """Форматирует список Utterance в читаемый текст."""
    lines = []
    for u in utterances:
        lines.append(f"[{u.start_time:.2f}s - {u.end_time:.2f}s] (conf: {u.confidence:.2f})")
        lines.append(f"  {u.text}")
        lines.append("")
    return "\n".join(lines)


def test_single_file(audio_path: Path, model_size: str = "base"):
    """
    Тестирует ASR на одном файле.
    
    Args:
        audio_path: Путь к аудио файлу
        model_size: Размер модели Whisper
    """
    print(f"Загрузка аудио: {audio_path}")
    audio = load_audio(audio_path)
    duration = len(audio) / 16000
    print(f"Длительность: {duration:.2f} секунд")
    print()
    
    print(f"Инициализация ASR модели ({model_size})...")
    asr = FasterWhisperEngine(
        model_size=model_size,
        device="cpu",
        compute_type="int8",
        cpu_threads=4
    )
    print("✓ Модель загружена")
    print()
    
    print("Распознавание речи...")
    utterances = asr.transcribe(audio)
    print(f"✓ Распознано {len(utterances)} сегментов")
    print()
    
    # Выводим результаты
    print("=" * 80)
    print("РЕЗУЛЬТАТЫ РАСПОЗНАВАНИЯ:")
    print("=" * 80)
    print(format_utterances(utterances))
    
    # Полный текст
    full_text = " ".join(u.text for u in utterances)
    print("=" * 80)
    print("ПОЛНЫЙ ТЕКСТ:")
    print("=" * 80)
    print(full_text)
    print()
    
    # Сохраняем результат
    output_path = Path(__file__).parent.parent / "data" / "open_stt" / "test_results"
    output_path.mkdir(parents=True, exist_ok=True)
    
    result_file = output_path / f"{audio_path.stem}_result.txt"
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(f"Файл: {audio_path}\n")
        f.write(f"Длительность: {duration:.2f} секунд\n")
        f.write(f"Сегментов: {len(utterances)}\n\n")
        f.write("РЕЗУЛЬТАТЫ:\n")
        f.write("=" * 80 + "\n")
        f.write(format_utterances(utterances))
        f.write("\n" + "=" * 80 + "\n")
        f.write("ПОЛНЫЙ ТЕКСТ:\n")
        f.write("=" * 80 + "\n")
        f.write(full_text)
    
    print(f"✓ Результат сохранен: {result_file}")
    print()
    print("СЛЕДУЮЩИЕ ШАГИ:")
    print("1. Прослушайте аудио файл")
    print(f"2. Сравните с результатом в {result_file}")
    print("3. Оцените точность распознавания")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Тестирование ASR на одном файле")
    parser.add_argument("audio_path", type=Path, help="Путь к аудио файлу")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium"],
                       help="Размер модели Whisper")
    
    args = parser.parse_args()
    
    if not args.audio_path.exists():
        print(f"Ошибка: файл не найден: {args.audio_path}")
        sys.exit(1)
    
    test_single_file(args.audio_path, args.model)

