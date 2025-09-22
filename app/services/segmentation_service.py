import os
import numpy as np
import librosa
import soundfile as sf
import logging
import torch
import shutil
import threading
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager
from app.tasks.notify_tasks import notify_progress_task

logger = logging.getLogger(__name__)


class SegmentationError(Exception):
    """Кастомное исключение для ошибок сегментации"""
    pass


class SileroVADManager:
    """Thread-safe singleton для управления Silero VAD моделью"""
    _instance = None
    _lock = threading.Lock()
    _model = None
    _utils = None

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        model, utils = torch.hub.load(
                            repo_or_dir='snakers4/silero-vad',
                            model='silero_vad',
                            force_reload=False
                        )
                        self._model = model
                        self._utils = utils
                        logger.info("Silero VAD модель загружена успешно")
                    except Exception as e:
                        raise SegmentationError(f"Ошибка загрузки Silero VAD: {e}")
        return self._utils[0], self._model  # get_speech_timestamps, model


@contextmanager
def safe_file_operations(output_dir: str, cleanup_on_error: bool = True):
    """Context manager для безопасных файловых операций"""
    created_files = []
    try:
        yield created_files
    except Exception as e:
        if cleanup_on_error:
            # Очищаем созданные файлы при ошибке
            for filepath in created_files:
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                        logger.info(f"Удален файл при cleanup: {filepath}")
                except Exception as cleanup_error:
                    logger.error(f"Ошибка cleanup файла {filepath}: {cleanup_error}")
        raise e


def validate_audio_parameters(min_length: float, max_length: float, min_silence_duration: float) -> None:
    """Валидация параметров сегментации"""
    if min_length <= 0:
        raise SegmentationError("min_length должен быть положительным")
    if max_length <= min_length:
        raise SegmentationError("max_length должен быть больше min_length")
    if min_silence_duration < 0:
        raise SegmentationError("min_silence_duration не может быть отрицательным")
    if min_length > 60 or max_length > 300:
        raise SegmentationError("Слишком большие значения длительности сегментов")


def load_and_validate_audio(input_wav_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int, float]:
    """Безопасная загрузка и валидация аудио файла"""
    if not os.path.exists(input_wav_path):
        raise SegmentationError(f"Входной файл не найден: {input_wav_path}")
    
    try:
        # Получаем размер файла для проверки
        file_size = os.path.getsize(input_wav_path)
        if file_size == 0:
            raise SegmentationError("Входной файл пуст")
        
        # Загружаем аудио
        y, sr = librosa.load(input_wav_path, sr=target_sr, mono=True)
        
        if len(y) == 0:
            raise SegmentationError("Аудио файл не содержит данных")
        
        if sr != target_sr:
            raise SegmentationError(f"Не удалось привести к частоте {target_sr}Hz")
        
        total_duration = len(y) / sr
        if total_duration < 1.0:
            raise SegmentationError("Слишком короткое аудио (менее 1 секунды)")
            
        if total_duration > 3600:  # 1 час
            raise SegmentationError("Слишком длинное аудио (более 1 часа)")
        
        logger.info(f"Аудио загружено: {total_duration:.2f}s, {sr}Hz, {len(y)} сэмплов")
        return y, sr, total_duration
        
    except librosa.LibrosaError as e:
        raise SegmentationError(f"Ошибка загрузки аудио с librosa: {e}")
    except Exception as e:
        raise SegmentationError(f"Неожиданная ошибка загрузки аудио: {e}")


def perform_vad_analysis(audio_data: np.ndarray, min_silence_duration: float, speech_pad: float) -> List[Dict]:
    """Выполнение VAD анализа с обработкой ошибок"""
    try:
        vad_manager = SileroVADManager()
        get_speech_timestamps, model = vad_manager.get_model()
        
        # Подготавливаем тензор
        wav_tensor = torch.from_numpy(audio_data).float()
        if wav_tensor.dim() == 1:
            wav_tensor = wav_tensor.unsqueeze(0)
        
        # Выполняем VAD
        speech_segments = get_speech_timestamps(
            wav_tensor,
            model,
            threshold=0.5,
            min_silence_duration_ms=int(min_silence_duration * 1000),
            speech_pad_ms=int(speech_pad * 1000)
        )
        
        # Конвертируем в удобный формат
        active_intervals = [
            {"start": s["start"] / 16000.0, "end": s["end"] / 16000.0}
            for s in speech_segments
        ]
        
        logger.info(f"VAD обнаружил {len(active_intervals)} речевых сегментов")
        return active_intervals
        
    except Exception as e:
        raise SegmentationError(f"Ошибка VAD анализа: {e}")


def merge_speech_intervals(intervals: List[Dict], min_silence_duration: float) -> List[Dict]:
    """Слияние близких речевых интервалов"""
    if not intervals:
        return []
    
    merged = []
    current = intervals[0].copy()
    
    for next_seg in intervals[1:]:
        if next_seg["start"] - current["end"] < min_silence_duration:
            current["end"] = next_seg["end"]
        else:
            merged.append(current)
            current = next_seg.copy()
    
    merged.append(current)
    logger.info(f"После слияния: {len(merged)} интервалов")
    return merged


def create_audio_segments(
    audio_data: np.ndarray,
    sr: int,
    merged_intervals: List[Dict],
    output_dir: str,
    min_length: float,
    max_length: float,
    max_extension: float,
    allow_short_final: bool,
    min_silence_duration: float,
    dataset_id: int
) -> Dict:
    """Создание аудио сегментов с обработкой ошибок"""
    
    if not os.path.exists(output_dir):
        raise SegmentationError(f"Выходная директория не существует: {output_dir}")
    
    total_duration = len(audio_data) / sr
    chunks = []
    durations = []
    current_time = 0.0
    split_points = []
    estimated_segments = max(1, int(total_duration / max_length))
    
    with safe_file_operations(output_dir) as created_files:
        segment_count = 0
        
        while current_time < total_duration:
            try:
                start_time = current_time
                min_end_time = start_time + min_length
                max_end_time = start_time + max_length
                hard_end_time = min(max_end_time, total_duration)
                search_end_time = min(max_end_time + max_extension, total_duration)

                # Поиск точки разреза
                cut_time = None
                for interval in merged_intervals:
                    if interval["start"] > min_end_time and interval["start"] <= search_end_time:
                        prev_end = current_time
                        for prev in merged_intervals:
                            if prev["end"] < interval["start"]:
                                prev_end = max(prev_end, prev["end"])
                        if interval["start"] - prev_end >= min_silence_duration:
                            cut_time = interval["start"]
                            break

                if cut_time is None:
                    cut_time = hard_end_time

                seg_duration = cut_time - start_time
                
                # Обработка коротких сегментов
                if seg_duration < min_length:
                    if cut_time >= total_duration and allow_short_final:
                        cut_time = total_duration
                    else:
                        cut_time = min(start_time + max_length, total_duration)
                        if cut_time - start_time < min_length:
                            if allow_short_final and start_time < total_duration:
                                cut_time = total_duration
                            else:
                                break

                # Проверка минимальной длительности
                if cut_time <= start_time + 1e-3:
                    cut_time = min(start_time + 0.2, total_duration)
                    if cut_time <= start_time:
                        break

                # Извлечение сегмента
                start_sample = int(start_time * sr)
                end_sample = int(cut_time * sr)
                
                if start_sample >= end_sample or end_sample > len(audio_data):
                    logger.warning(f"Некорректные границы сегмента: {start_sample}-{end_sample}")
                    break
                    
                segment = audio_data[start_sample:end_sample]

                if len(segment) == 0:
                    logger.warning("Пустой сегмент, прерываем")
                    break

                # Сохранение сегмента
                filename = f"segment_{segment_count + 1:04d}.wav"
                filepath = os.path.join(output_dir, filename)
                
                try:
                    sf.write(filepath, segment, sr, subtype='PCM_16')
                    created_files.append(filepath)
                    
                    # Проверяем что файл действительно создался и не пуст
                    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                        raise SegmentationError(f"Файл не был создан или пуст: {filepath}")
                        
                except Exception as e:
                    raise SegmentationError(f"Ошибка записи файла {filepath}: {e}")

                chunks.append({"start": start_time, "end": cut_time, "samples": segment})
                durations.append(cut_time - start_time)
                split_points.append(cut_time)
                segment_count += 1

                # Уведомление о прогрессе
                progress = 60 + int((segment_count / estimated_segments) * 30)
                progress = min(progress, 90)
                notify_progress_task.delay(dataset_id=dataset_id, task="Формирование сегментов", progress=progress)

                current_time = cut_time
                if current_time >= total_duration - 1e-3:
                    break
                    
            except Exception as e:
                logger.error(f"Ошибка создания сегмента {segment_count + 1}: {e}")
                raise SegmentationError(f"Ошибка создания сегмента: {e}")

        if segment_count == 0:
            raise SegmentationError("Не удалось создать ни одного сегмента")

    return {
        "chunks": chunks,
        "durations": durations,
        "split_points": split_points,
        "segments_count": segment_count
    }


def segment_audio(
    input_wav_path: str,
    output_dir: str,
    min_length: float = 1.5,
    max_length: float = 12.0,
    min_silence_duration: float = 0.3,
    speech_pad: float = 0.05,
    frame_length: int = 512,
    hop_length: int = 160,
    max_extension: float = 1.5,
    allow_short_final: bool = True,
    debug: bool = False,
    dataset_id: int = 1
) -> Dict:
    """Главная функция сегментации с полной обработкой ошибок"""
    
    try:
        # Валидация параметров
        validate_audio_parameters(min_length, max_length, min_silence_duration)
        
        # Создаем выходную директорию
        os.makedirs(output_dir, exist_ok=True)
        
        # Шаг 1: Загрузка WAV
        notify_progress_task.delay(dataset_id=dataset_id, task="Загрузка WAV", progress=0)
        y, sr, total_duration = load_and_validate_audio(input_wav_path)
        notify_progress_task.delay(dataset_id=dataset_id, task="Загрузка WAV", progress=10)

        # Шаг 2: VAD анализ
        notify_progress_task.delay(dataset_id=dataset_id, task="VAD анализ", progress=15)
        active_intervals = perform_vad_analysis(y, min_silence_duration, speech_pad)
        notify_progress_task.delay(dataset_id=dataset_id, task="VAD анализ", progress=30)

        if not active_intervals:
            return {
                "status": "warning", 
                "message": "Речь не обнаружена", 
                "segments_count": 0,
                "stats": {"total_chunks": 0, "saved": 0}
            }

        # Шаг 3: Слияние интервалов
        notify_progress_task.delay(dataset_id=dataset_id, task="Сегментация речи", progress=40)
        merged = merge_speech_intervals(active_intervals, min_silence_duration)
        notify_progress_task.delay(dataset_id=dataset_id, task="Сегментация речи", progress=50)

        # Шаг 4: Создание сегментов
        result = create_audio_segments(
            y, sr, merged, output_dir, min_length, max_length, 
            max_extension, allow_short_final, min_silence_duration, dataset_id
        )
        
        # Шаг 5: Финализация
        notify_progress_task.delay(dataset_id=dataset_id, task="Запись файлов", progress=100)

        stats = {
            "total_chunks": len(result["chunks"]),
            "saved": result["segments_count"],
            "avg_duration": float(np.mean(result["durations"])) if result["durations"] else 0,
            "min_duration": float(np.min(result["durations"])) if result["durations"] else 0,
            "max_duration": float(np.max(result["durations"])) if result["durations"] else 0,
            "allow_short_final": allow_short_final,
            "total_input_duration": total_duration
        }

        logger.info(f"Сегментация завершена: {result['segments_count']} сегментов")
        
        return {
            "status": "success",
            "segments_count": result["segments_count"],
            "stats": stats,
            "split_points": result["split_points"]
        }

    except SegmentationError as e:
        logger.error(f"Ошибка сегментации: {e}")
        # Очищаем выходную директорию при ошибке
        try:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
                logger.info(f"Очищена директория после ошибки: {output_dir}")
        except Exception as cleanup_error:
            logger.error(f"Ошибка очистки директории: {cleanup_error}")
        
        return {
            "status": "error",
            "message": str(e),
            "segments_count": 0,
            "stats": {"total_chunks": 0, "saved": 0}
        }
        
    except Exception as e:
        logger.exception(f"Неожиданная ошибка сегментации: {e}")
        # Очищаем выходную директорию при ошибке
        try:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
                logger.info(f"Очищена директория после неожиданной ошибки: {output_dir}")
        except Exception as cleanup_error:
            logger.error(f"Ошибка очистки директории: {cleanup_error}")
            
        return {
            "status": "error",
            "message": f"Неожиданная ошибка: {str(e)}",
            "segments_count": 0,
            "stats": {"total_chunks": 0, "saved": 0}
        }