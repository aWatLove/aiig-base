"""
Grid Search Benchmark для Voice-to-Text на MacBook Air M4.

Автоматический бенчмарк для поиска оптимальной конфигурации faster-whisper
с учетом скорости и точности.
"""

import csv
import gc
import sys
import time
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import psutil
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "v2t_service" / "src"))

from v2t.asr.whisper_impl import FasterWhisperEngine


def load_audio(path: Path, target_sr: int = 16000) -> Tuple[np.ndarray, float]:
    """
    Загружает аудио файл и приводит к нужному формату.
    
    Args:
        path: Путь к аудио файлу
        target_sr: Целевая частота дискретизации
        
    Returns:
        Кортеж (аудио данные, длительность в секундах)
    """
    try:
        audio, sr = sf.read(str(path))
    except:
        import librosa
        audio, sr = librosa.load(str(path), sr=None, mono=False)
    
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    duration = len(audio) / target_sr
    return audio.astype(np.float32), duration


def load_audio_files(data_dir: Path, max_files: int = 20) -> List[Tuple[Path, np.ndarray, float]]:
    """
    Загружает тестовые аудио файлы.
    
    Args:
        data_dir: Директория с аудио файлами
        max_files: Максимальное количество файлов
        
    Returns:
        Список кортежей (путь, аудио данные, длительность)
    """
    audio_extensions = {'.wav', '.mp3', '.opus', '.flac', '.m4a', '.ogg'}
    audio_files = [
        f for f in data_dir.rglob('*')
        if f.is_file() and f.suffix.lower() in audio_extensions and 'synthetic' not in str(f)
    ]
    
    if len(audio_files) < max_files:
        print(f"⚠ Найдено только {len(audio_files)} файлов, требуется {max_files}")
        max_files = len(audio_files)
    
    selected_files = audio_files[:max_files]
    print(f"Загрузка {len(selected_files)} тестовых файлов...")
    
    loaded_files = []
    for file_path in tqdm(selected_files, desc="Загрузка аудио"):
        try:
            audio, duration = load_audio(file_path)
            loaded_files.append((file_path, audio, duration))
        except Exception as e:
            print(f"⚠ Ошибка загрузки {file_path.name}: {e}")
    
    total_duration = sum(d for _, _, d in loaded_files)
    print(f"✓ Загружено {len(loaded_files)} файлов, общая длительность: {total_duration:.2f} сек ({total_duration/60:.2f} мин)\n")
    
    return loaded_files


def generate_grid_combinations() -> List[Dict]:
    """
    Генерирует все комбинации параметров для grid search.
    
    Returns:
        Список словарей с параметрами конфигураций
    """
    model_sizes = ["tiny", "base", "small", "medium"]
    compute_types = ["int8", "float32"]
    cpu_threads_list = [2, 4, 8]
    beam_sizes = [1, 2, 5]
    vad_filters = [True, False]
    
    combinations = []
    for model_size, compute_type, cpu_threads, beam_size, vad_filter in product(
        model_sizes, compute_types, cpu_threads_list, beam_sizes, vad_filters
    ):
        combinations.append({
            "model_size": model_size,
            "compute_type": compute_type,
            "cpu_threads": cpu_threads,
            "beam_size": beam_size,
            "vad_filter": vad_filter
        })
    
    return combinations


def test_configuration(
    config: Dict,
    audio_files: List[Tuple[Path, np.ndarray, float]]
) -> Dict:
    """
    Тестирует одну конфигурацию параметров.
    
    Args:
        config: Словарь с параметрами конфигурации
        audio_files: Список загруженных аудио файлов
        
    Returns:
        Словарь с результатами тестирования
    """
    result = {
        **config,
        "init_time": 0.0,
        "total_processing_time": 0.0,
        "total_audio_duration": 0.0,
        "rtf": 0.0,
        "avg_confidence": 0.0,
        "peak_memory_mb": 0.0,
        "success": False,
        "error": None
    }
    
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024 / 1024
    
    try:
        init_start = time.time()
        engine = FasterWhisperEngine(
            model_size=config["model_size"],
            device="cpu",
            compute_type=config["compute_type"],
            cpu_threads=config["cpu_threads"]
        )
        init_time = time.time() - init_start
        result["init_time"] = init_time
        
        total_duration = 0.0
        total_processing_time = 0.0
        all_confidences = []
        peak_memory = memory_before
        
        for file_path, audio, duration in audio_files:
            total_duration += duration
            
            process_start = time.time()
            utterances = engine.transcribe(
                audio,
                beam_size=config["beam_size"],
                vad_filter=config["vad_filter"]
            )
            process_time = time.time() - process_start
            total_processing_time += process_time
            
            if utterances:
                confidences = [u.confidence for u in utterances]
                all_confidences.extend(confidences)
            
            current_memory = process.memory_info().rss / 1024 / 1024
            peak_memory = max(peak_memory, current_memory)
        
        result["total_audio_duration"] = total_duration
        result["total_processing_time"] = total_processing_time
        result["rtf"] = total_processing_time / total_duration if total_duration > 0 else 0.0
        result["avg_confidence"] = np.mean(all_confidences) if all_confidences else 0.0
        result["peak_memory_mb"] = peak_memory - memory_before
        result["success"] = True
        
        del engine
        gc.collect()
        
    except Exception as e:
        result["error"] = str(e)
        result["success"] = False
        gc.collect()
    
    return result


def save_results_csv(results: List[Dict], filename: Path):
    """
    Сохраняет результаты в CSV файл.
    
    Args:
        results: Список результатов тестирования
        filename: Путь к файлу для сохранения
    """
    if not results:
        return
    
    fieldnames = [
        "model_size", "compute_type", "cpu_threads", "beam_size", "vad_filter",
        "init_time", "total_processing_time", "total_audio_duration", "rtf",
        "avg_confidence", "peak_memory_mb", "success", "error"
    ]
    
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✓ Результаты сохранены в CSV: {filename}")


def generate_markdown_report(results: List[Dict], filename: Path):
    """
    Генерирует Markdown отчет с таблицей результатов.
    
    Args:
        results: Список результатов тестирования
        filename: Путь к файлу для сохранения
    """
    successful = [r for r in results if r["success"]]
    
    if not successful:
        with open(filename, "w", encoding="utf-8") as f:
            f.write("# Grid Search Results\n\n")
            f.write("❌ Нет успешных результатов тестирования.\n")
        return
    
    successful.sort(key=lambda x: x["rtf"])
    
    lines = []
    lines.append("# Grid Search ASR Benchmark Results\n")
    lines.append(f"**Всего конфигураций:** {len(results)}\n")
    lines.append(f"**Успешных:** {len(successful)}\n")
    lines.append(f"**Неудачных:** {len(results) - len(successful)}\n\n")
    
    lines.append("## Топ-10 конфигураций по скорости (RTF)\n")
    lines.append("| Rank | Model | Compute | Threads | Beam | VAD | RTF | Confidence | Memory (MB) | Init (s) |\n")
    lines.append("|------|-------|---------|---------|------|-----|-----|------------|-------------|----------|\n")
    
    for i, result in enumerate(successful[:10], 1):
        lines.append(
            f"| {i} | {result['model_size']} | {result['compute_type']} | "
            f"{result['cpu_threads']} | {result['beam_size']} | "
            f"{'✓' if result['vad_filter'] else '✗'} | "
            f"{result['rtf']:.4f} | {result['avg_confidence']:.3f} | "
            f"{result['peak_memory_mb']:.1f} | {result['init_time']:.2f} |\n"
        )
    
    lines.append("\n## Статистика по моделям\n")
    lines.append("| Model | Avg RTF | Best RTF | Avg Confidence | Avg Memory (MB) |\n")
    lines.append("|-------|---------|----------|----------------|-----------------|\n")
    
    for model_size in ["tiny", "base", "small", "medium"]:
        model_results = [r for r in successful if r["model_size"] == model_size]
        if model_results:
            avg_rtf = np.mean([r["rtf"] for r in model_results])
            best_rtf = min([r["rtf"] for r in model_results])
            avg_conf = np.mean([r["avg_confidence"] for r in model_results])
            avg_mem = np.mean([r["peak_memory_mb"] for r in model_results])
            
            lines.append(
                f"| {model_size} | {avg_rtf:.4f} | {best_rtf:.4f} | "
                f"{avg_conf:.3f} | {avg_mem:.1f} |\n"
            )
    
    failed = [r for r in results if not r["success"]]
    if failed:
        lines.append("\n## Ошибки\n")
        lines.append("| Model | Compute | Threads | Beam | VAD | Error |\n")
        lines.append("|-------|---------|---------|------|-----|-------|\n")
        
        for result in failed[:10]:
            error_msg = result["error"][:50] + "..." if len(result["error"]) > 50 else result["error"]
            lines.append(
                f"| {result['model_size']} | {result['compute_type']} | "
                f"{result['cpu_threads']} | {result['beam_size']} | "
                f"{'✓' if result['vad_filter'] else '✗'} | {error_msg} |\n"
            )
    
    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(lines)
    
    print(f"✓ Markdown отчет сохранен: {filename}")


def main():
    """Основная функция grid search."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Grid Search Benchmark для ASR")
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Директория с аудио файлами"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=20,
        help="Максимальное количество тестовых файлов (по умолчанию 20)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Директория для сохранения результатов (по умолчанию data/open_stt/test_results)"
    )
    
    args = parser.parse_args()
    
    if not args.data_dir.exists():
        print(f"❌ Ошибка: директория не найдена: {args.data_dir}")
        sys.exit(1)
    
    # Определяем директорию для результатов
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = Path(__file__).parent.parent / "data" / "open_stt" / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("GRID SEARCH ASR BENCHMARK")
    print("=" * 80)
    print(f"Директория с данными: {args.data_dir}")
    print(f"Количество тестовых файлов: {args.max_files}")
    print(f"Директория результатов: {output_dir}\n")
    
    audio_files = load_audio_files(args.data_dir, args.max_files)
    
    if not audio_files:
        print("❌ Не найдено аудио файлов для тестирования")
        sys.exit(1)
    
    combinations = generate_grid_combinations()
    print(f"Всего комбинаций для тестирования: {len(combinations)}\n")
    
    results = []
    print("Начало grid search...\n")
    
    for config in tqdm(combinations, desc="Grid Search"):
        config_str = (
            f"{config['model_size']}/{config['compute_type']}/"
            f"threads={config['cpu_threads']}/beam={config['beam_size']}/"
            f"vad={config['vad_filter']}"
        )
        
        result = test_configuration(config, audio_files)
        results.append(result)
        
        if len(results) % 10 == 0:
            save_results_csv(results, output_dir / "grid_search_results_intermediate.csv")
    
    csv_file = output_dir / "grid_search_results.csv"
    save_results_csv(results, csv_file)
    
    md_file = output_dir / "grid_search_report.md"
    generate_markdown_report(results, md_file)
    
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    print("\n" + "=" * 80)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 80)
    print(f"Всего конфигураций: {len(results)}")
    print(f"Успешных: {len(successful)}")
    print(f"Неудачных: {len(failed)}")
    
    if successful:
        best = min(successful, key=lambda x: x["rtf"])
        print(f"\n🏆 Лучшая конфигурация (RTF={best['rtf']:.4f}):")
        print(f"   Model: {best['model_size']}")
        print(f"   Compute: {best['compute_type']}")
        print(f"   Threads: {best['cpu_threads']}")
        print(f"   Beam: {best['beam_size']}")
        print(f"   VAD: {best['vad_filter']}")
        print(f"   Confidence: {best['avg_confidence']:.3f}")
        print(f"   Memory: {best['peak_memory_mb']:.1f} MB")
    
    print(f"\n✓ Результаты сохранены:")
    print(f"   CSV: {csv_file}")
    print(f"   Markdown: {md_file}")


if __name__ == "__main__":
    main()

