"""
Скрипт для батч-тестирования ASR на ~1 ГБ данных (2 итерация).

Проверяет точность распознавания на множестве файлов.
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Dict

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "v2t_service" / "src"))

from v2t.asr.whisper_impl import FasterWhisperEngine
from v2t.core.dto import Utterance


def load_audio(path: Path, target_sr: int = 16000) -> np.ndarray:
    """Загружает аудио файл."""
    try:
        audio, sr = sf.read(str(path))
    except:
        audio, sr = librosa.load(str(path), sr=None, mono=False)
    
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    return audio.astype(np.float32)


def process_file(asr: FasterWhisperEngine, audio_path: Path) -> Dict:
    """
    Обрабатывает один файл и возвращает метрики.
    
    Args:
        asr: ASR движок
        audio_path: Путь к аудио файлу
        
    Returns:
        Словарь с результатами обработки
    """
    try:
        # Загрузка
        audio = load_audio(audio_path)
        duration = len(audio) / 16000
        
        # Распознавание
        start_time = time.time()
        utterances = asr.transcribe(audio)
        processing_time = time.time() - start_time
        
        # Метрики
        full_text = " ".join(u.text for u in utterances)
        avg_confidence = np.mean([u.confidence for u in utterances]) if utterances else 0.0
        
        return {
            "file": str(audio_path),
            "duration": duration,
            "processing_time": processing_time,
            "rtf": processing_time / duration if duration > 0 else 0.0,
            "segments_count": len(utterances),
            "avg_confidence": avg_confidence,
            "full_text": full_text,
            "success": True,
            "error": None
        }
    except Exception as e:
        return {
            "file": str(audio_path),
            "success": False,
            "error": str(e)
        }


def test_batch(data_dir: Path, model_size: str = "base", max_files: int = None):
    """
    Тестирует ASR на всех файлах в директории.
    
    Args:
        data_dir: Директория с аудио файлами
        model_size: Размер модели Whisper
        max_files: Максимальное количество файлов для обработки
    """
    # Находим все аудио файлы
    audio_extensions = {'.wav', '.mp3', '.opus', '.flac', '.m4a', '.ogg'}
    audio_files = [
        f for f in data_dir.rglob('*')
        if f.suffix.lower() in audio_extensions and f.is_file()
    ]
    
    if not audio_files:
        print(f"Не найдено аудио файлов в {data_dir}")
        return
    
    if max_files:
        audio_files = audio_files[:max_files]
    
    print(f"Найдено файлов: {len(audio_files)}")
    print(f"Инициализация ASR модели ({model_size})...")
    
    asr = FasterWhisperEngine(
        model_size=model_size,
        device="cpu",
        compute_type="int8",
        cpu_threads=4
    )
    print("✓ Модель загружена\n")
    
    # Обработка файлов
    results = []
    total_duration = 0.0
    total_processing_time = 0.0
    
    for audio_file in tqdm(audio_files, desc="Обработка"):
        result = process_file(asr, audio_file)
        results.append(result)
        
        if result["success"]:
            total_duration += result["duration"]
            total_processing_time += result["processing_time"]
    
    # Статистика
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ БАТЧ-ТЕСТИРОВАНИЯ")
    print("=" * 80)
    print(f"Всего файлов: {len(results)}")
    print(f"Успешно обработано: {len(successful)}")
    print(f"Ошибок: {len(failed)}")
    
    if successful:
        avg_rtf = np.mean([r["rtf"] for r in successful])
        avg_confidence = np.mean([r["avg_confidence"] for r in successful])
        total_segments = sum(r["segments_count"] for r in successful)
        
        print(f"\nОбщая длительность аудио: {total_duration:.2f} секунд ({total_duration/3600:.2f} часов)")
        print(f"Общее время обработки: {total_processing_time:.2f} секунд ({total_processing_time/3600:.2f} часов)")
        print(f"Средний RTF: {avg_rtf:.3f}")
        print(f"Средняя уверенность: {avg_confidence:.3f}")
        print(f"Всего сегментов: {total_segments}")
        
        if total_duration > 0:
            overall_rtf = total_processing_time / total_duration
            print(f"Общий RTF: {overall_rtf:.3f}")
    
    # Сохранение результатов
    output_dir = Path(__file__).parent.parent / "data" / "open_stt" / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "batch_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Результаты сохранены: {results_file}")
    
    # Сохраняем статистику
    stats_file = output_dir / "batch_stats.txt"
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write("СТАТИСТИКА БАТЧ-ТЕСТИРОВАНИЯ\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Всего файлов: {len(results)}\n")
        f.write(f"Успешно: {len(successful)}\n")
        f.write(f"Ошибок: {len(failed)}\n\n")
        
        if successful:
            f.write(f"Общая длительность: {total_duration:.2f} сек ({total_duration/3600:.2f} ч)\n")
            f.write(f"Общее время обработки: {total_processing_time:.2f} сек ({total_processing_time/3600:.2f} ч)\n")
            f.write(f"Средний RTF: {avg_rtf:.3f}\n")
            f.write(f"Общий RTF: {overall_rtf:.3f}\n")
            f.write(f"Средняя уверенность: {avg_confidence:.3f}\n")
            f.write(f"Всего сегментов: {total_segments}\n")
    
    print(f"✓ Статистика сохранена: {stats_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Батч-тестирование ASR")
    parser.add_argument("data_dir", type=Path, help="Директория с аудио файлами")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium"],
                       help="Размер модели Whisper")
    parser.add_argument("--max-files", type=int, help="Максимальное количество файлов")
    
    args = parser.parse_args()
    
    if not args.data_dir.exists():
        print(f"Ошибка: директория не найдена: {args.data_dir}")
        sys.exit(1)
    
    test_batch(args.data_dir, args.model, args.max_files)

