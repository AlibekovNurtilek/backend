import os
import numpy as np
import librosa
import soundfile as sf
import logging
import shutil
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager
from pathlib import Path
from app.tasks.notify_tasks import notify_progress_task

logger = logging.getLogger(__name__)


class SegmentationError(Exception):
    """Кастомное исключение для ошибок сегментации"""
    pass


class ImprovedAudioSegmenter:
    """Новый алгоритм сегментации на основе энергетического анализа"""
    
    def __init__(self, 
                 min_duration=5.0, 
                 max_duration=25.0, 
                 split_coefficient=2,
                 initial_silence_threshold=0.005,
                 max_silence_threshold=0.05,
                 max_discard_ratio=0.1,
                 min_silence_duration=0.1):
        """
        Args:
            min_duration: Минимальная длительность сегмента (сек)
            max_duration: Максимальная длительность сегмента (сек)
            split_coefficient: Коэффициент k для первичного деления
            initial_silence_threshold: Начальный порог тишины
            max_silence_threshold: Минимальный порог тишины
            max_discard_ratio: Максимальный процент аудио для выброса (0.1 = 10%)
            min_silence_duration: Минимальная длительность тишины для деления (сек)
        """
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.split_coefficient = split_coefficient
        self.initial_silence_threshold = initial_silence_threshold
        self.max_silence_threshold = max_silence_threshold
        self.max_discard_ratio = max_discard_ratio
        self.min_silence_duration = min_silence_duration
        
        # Мемоизация энергий окон
        self.window_energies = None
        self.window_positions = None
        self.sr = None
        self.window_size = None
        self.hop_size = None
        
        # Статистика
        self.total_segments = 0
        self.discarded_segments = 0
        self.total_duration = 0.0
        self.discarded_duration = 0.0

    def segment_audio(self, audio_path: str, output_dir: str, dataset_id: int = 1) -> Dict:
        """Сегментирует wav-файл и сохраняет сегменты в output_dir"""
        logger.info(f"Начинаем сегментацию: {audio_path}")
        
        try:
            # Загрузка аудио
            notify_progress_task.delay(dataset_id=dataset_id, task="Загрузка аудио", progress=10)
            audio_np, sr = librosa.load(audio_path, sr=None)
            self.sr = sr
            self.total_duration = len(audio_np) / sr
            
            if len(audio_np) == 0:
                raise SegmentationError("Аудио файл пуст")
            
            # Нормализация
            if np.max(np.abs(audio_np)) > 0:
                normalized_audio = audio_np / np.max(np.abs(audio_np))
            else:
                normalized_audio = audio_np
                
            notify_progress_task.delay(dataset_id=dataset_id, task="Анализ энергии", progress=25)
            
            # Мемоизация - один проход для вычисления всех энергий окон
            self._compute_window_energies(normalized_audio)
            
            notify_progress_task.delay(dataset_id=dataset_id, task="Рекурсивное деление", progress=40)
            
            # Рекурсивная сегментация
            segments = self._recursive_split(0, len(audio_np))
            
            notify_progress_task.delay(dataset_id=dataset_id, task="Фильтрация сегментов", progress=60)
            
            # Фильтруем по длительности
            final_segments = self._filter_segments(segments)
            
            if not final_segments:
                raise SegmentationError("Не удалось создать валидные сегменты")
            
            notify_progress_task.delay(dataset_id=dataset_id, task="Экспорт сегментов", progress=75)
            
            # Экспортируем
            segment_info = self._export_segments(final_segments, audio_np, sr, output_dir, dataset_id)
            
            notify_progress_task.delay(dataset_id=dataset_id, task="Сегментация завершена", progress=100)
            
            # Подготовка статистики
            durations = [seg['duration'] for seg in segment_info]
            stats = {
                "total_chunks": len(segment_info),
                "saved": len(segment_info),
                "avg_duration": float(np.mean(durations)) if durations else 0,
                "min_duration": float(np.min(durations)) if durations else 0,
                "max_duration": float(np.max(durations)) if durations else 0,
                "total_segments": self.total_segments,
                "discarded_segments": self.discarded_segments,
                "total_input_duration": self.total_duration,
                "discarded_duration": self.discarded_duration
            }
            
            logger.info(f"Сегментация завершена: {len(segment_info)} сегментов")
            
            return {
                "status": "success",
                "segments_count": len(segment_info),
                "stats": stats
            }
            
        except Exception as e:
            logger.exception(f"Ошибка сегментации: {e}")
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

    def _compute_window_energies(self, audio: np.ndarray):
        """Вычисляет и сохраняет энергии всех окон для мемоизации"""
        logger.info("Вычисляем энергии окон для мемоизации...")
        
        # Параметры анализа
        self.window_size = max(int(0.04 * self.sr), 100)  # ~40мс окна
        self.hop_size = self.window_size // 4
        
        self.window_energies = []
        self.window_positions = []
        
        # RMS энергия для каждого окна
        for i in range(0, len(audio) - self.window_size, self.hop_size):
            window = audio[i:i + self.window_size]
            energy = np.sqrt(np.mean(window ** 2))
            
            self.window_energies.append(energy)
            self.window_positions.append(i + self.window_size // 2)
        
        logger.info(f"Вычислено {len(self.window_energies)} окон энергии")

    def _recursive_split(self, start_sample: int, end_sample: int, 
                        silence_threshold: Optional[float] = None) -> List[Tuple[int, int]]:
        if silence_threshold is None:
            silence_threshold = self.initial_silence_threshold
            
        segment_length = end_sample - start_sample
        duration_sec = segment_length / self.sr        
        if duration_sec <= self.max_duration:
            return [(start_sample, end_sample)]
        
        silence_intervals = self._find_silence_intervals(start_sample, end_sample, silence_threshold)
        
        if not silence_intervals:
            if silence_threshold <self.max_silence_threshold:
                new_threshold = min(silence_threshold * 2, self.max_silence_threshold)
                logger.info(f"Снижаем порог тишины: {silence_threshold:.4f} -> {new_threshold:.4f}")
                return self._recursive_split(start_sample, end_sample, new_threshold)
            else:
                logger.debug(f"Выбрасываем сегмент {duration_sec:.2f}с - не найдена тишина")
                self.discarded_segments += 1
                self.discarded_duration += duration_sec
                return []
        
        best_silence = self._select_best_silence(silence_intervals, start_sample, end_sample)
        silence_start, silence_end = best_silence
        
        split_point = (silence_start + silence_end) // 2
        
        logger.debug(f"Делим в точке {split_point} (тишина {silence_start}-{silence_end})")
        
        left_segments = self._recursive_split(start_sample, split_point, silence_threshold)
        right_segments = self._recursive_split(split_point, end_sample, silence_threshold)
        
        return left_segments + right_segments

    def _find_silence_intervals(self, start_sample: int, end_sample: int, 
                               silence_threshold: float) -> List[Tuple[int, int]]:
        start_window_idx = None
        end_window_idx = None
        
        for i, pos in enumerate(self.window_positions):
            if start_window_idx is None and pos >= start_sample:
                start_window_idx = i
            if pos <= end_sample:
                end_window_idx = i
        
        if start_window_idx is None or end_window_idx is None:
            return []
        
        silence_windows = []
        for i in range(start_window_idx, end_window_idx + 1):
            if self.window_energies[i] < silence_threshold:
                silence_windows.append(i)
        
        if not silence_windows:
            return []
        
        silence_intervals = []
        current_start = silence_windows[0]
        current_end = silence_windows[0]
        
        for i in range(1, len(silence_windows)):
            window_idx = silence_windows[i]
            if window_idx == current_end + 1:
                current_end = window_idx
            else:
                interval_start = self.window_positions[current_start] - self.window_size // 2
                interval_end = self.window_positions[current_end] + self.window_size // 2
                silence_intervals.append((interval_start, interval_end))
                current_start = window_idx
                current_end = window_idx
        
        interval_start = self.window_positions[current_start] - self.window_size // 2
        interval_end = self.window_positions[current_end] + self.window_size // 2
        silence_intervals.append((interval_start, interval_end))
        
        min_silence_samples = int(self.min_silence_duration * self.sr)
        filtered_intervals = []
        
        for start, end in silence_intervals:
            silence_length = end - start
            if silence_length >= min_silence_samples:
                filtered_intervals.append((start, end))
            else:
                logger.debug(f"Отброшена короткая тишина: {silence_length/self.sr:.3f}с < {self.min_silence_duration}с")
        
        return filtered_intervals

    def _select_best_silence(self, silence_intervals: List[Tuple[int, int]], 
                            start_sample: int, end_sample: int) -> Tuple[int, int]:
        segment_center = (start_sample + end_sample) / 2
        
        best_silence = None
        best_length = 0
        best_distance_to_center = float('inf')
        
        for silence_start, silence_end in silence_intervals:
            silence_length = silence_end - silence_start
            silence_center = (silence_start + silence_end) / 2
            distance_to_center = abs(silence_center - segment_center)
            
            is_better = (silence_length > best_length or 
                        (silence_length == best_length and distance_to_center < best_distance_to_center))
            
            if is_better:
                best_silence = (silence_start, silence_end)
                best_length = silence_length
                best_distance_to_center = distance_to_center
        
        return best_silence

    def _filter_segments(self, segments: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not segments:
            return []
            
        filtered_segments = []
        min_samples = int(self.min_duration * self.sr)
        
        for start, end in segments:
            duration_samples = end - start
            duration_sec = duration_samples / self.sr
            
            if duration_samples < min_samples:
                logger.debug(f"Отбрасываем короткий сегмент: {duration_sec:.2f}с")
                self.discarded_segments += 1
                self.discarded_duration += duration_sec
                continue
            
            filtered_segments.append((start, end))
            self.total_segments += 1
        
        discarded_ratio = (self.discarded_duration / self.total_duration * 100 
                           if self.total_duration > 0 else 0)
        logger.info(f"Статистика: всего={self.total_segments}, "
                    f"отброшено={self.discarded_segments}, "
                    f"выброшено={discarded_ratio:.2f}% от общей длительности")
        
        return filtered_segments

    def _export_segments(self, segments: List[Tuple[int, int]], audio_np: np.ndarray, 
                        sr: int, output_dir: str, dataset_id: int) -> List[Dict]:
        segments_dir = Path(output_dir)
        if segments_dir.exists():
            shutil.rmtree(segments_dir)
        segments_dir.mkdir(parents=True, exist_ok=True)

        segment_info = []
        total_segments = len(segments)
        
        for i, (start_sample, end_sample) in enumerate(segments, 1):
            chunk_np = audio_np[start_sample:end_sample]
            duration_sec = len(chunk_np) / sr
            start_time_sec = start_sample / sr
            end_time_sec = end_sample / sr

            filename = f"segment_{i:04d}.wav"
            filepath = segments_dir / filename
            
            try:
                sf.write(str(filepath), chunk_np, sr, subtype='PCM_16')
                
                # Проверяем что файл действительно создался и не пуст
                if not filepath.exists() or filepath.stat().st_size == 0:
                    raise SegmentationError(f"Файл не был создан или пуст: {filepath}")
                    
            except Exception as e:
                raise SegmentationError(f"Ошибка записи файла {filepath}: {e}")

            segment_info.append({
                'path': str(filepath),
                'filename': filename,
                'start': start_time_sec,
                'end': end_time_sec,
                'duration': duration_sec,
                'index': i,
                'start_sample': start_sample,
                'end_sample': end_sample
            })
            
            # Уведомление о прогрессе экспорта
            if i % max(1, total_segments // 10) == 0:
                progress = 75 + int((i / total_segments) * 20)
                notify_progress_task.delay(dataset_id=dataset_id, task="Экспорт сегментов", progress=progress)

        logger.info(f"Сегментация завершена. Экспортировано {len(segment_info)} сегментов.")
        
        durations = [seg['duration'] for seg in segment_info]
        if durations:
            logger.info(f"Длительности: мин={min(durations):.2f}с, макс={max(durations):.2f}с, средн={np.mean(durations):.2f}с")
        
        return segment_info


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


def validate_audio_parameters(min_length: float, max_length: float) -> None:
    """Валидация параметров сегментации"""
    if min_length <= 0:
        raise SegmentationError("min_length должен быть положительным")
    if max_length <= min_length:
        raise SegmentationError("max_length должен быть больше min_length")
    if min_length > 60 or max_length > 300:
        raise SegmentationError("Слишком большие значения длительности сегментов")


def load_and_validate_audio(input_wav_path: str) -> Tuple[float]:
    """Быстрая валидация аудио файла"""
    if not os.path.exists(input_wav_path):
        raise SegmentationError(f"Входной файл не найден: {input_wav_path}")
    
    try:
        # Получаем размер файла для проверки
        file_size = os.path.getsize(input_wav_path)
        if file_size == 0:
            raise SegmentationError("Входной файл пуст")
        
        # Быстрая проверка длительности без полной загрузки
        duration = librosa.get_duration(filename=input_wav_path)
        
        if duration < 1.0:
            raise SegmentationError("Слишком короткое аудио (менее 1 секунды)")
            
        if duration > 36000:  # 10 часов
            raise SegmentationError("Слишком длинное аудио (более 10 часов)")
        
        logger.info(f"Аудио валидация пройдена: {duration:.2f}с")
        return duration
        
    except librosa.LibrosaError as e:
        raise SegmentationError(f"Ошибка валидации аудио с librosa: {e}")
    except Exception as e:
        raise SegmentationError(f"Неожиданная ошибка валидации аудио: {e}")


def segment_audio(
    input_wav_path: str,
    output_dir: str,
    min_length: float = 5,
    max_length: float = 25,
    min_silence_duration: float = 0.1,
    dataset_id: int = 1
) -> Dict:
    """Главная функция сегментации с новым алгоритмом"""
    
    try:
        # Валидация параметров
        validate_audio_parameters(min_length, max_length)
        
        # Создаем выходную директорию
        os.makedirs(output_dir, exist_ok=True)
        
        # Быстрая валидация файла
        notify_progress_task.delay(dataset_id=dataset_id, task="Валидация аудио", progress=5)
        total_duration = load_and_validate_audio(input_wav_path)
        
        # Создаем сегментатор с новыми параметрами
        segmenter = ImprovedAudioSegmenter(
            min_duration=min_length,
            max_duration=max_length,
            split_coefficient=2,
            initial_silence_threshold=0.005,
            max_silence_threshold=0.05,
            max_discard_ratio=0.1,
            min_silence_duration=min_silence_duration
        )
        
        # Выполняем сегментацию
        result = segmenter.segment_audio(input_wav_path, output_dir, dataset_id)
        
        if result['status'] == 'error':
            return result
        
        logger.info(f"Сегментация завершена: {result['segments_count']} сегментов")
        
        return result

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