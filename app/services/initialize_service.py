import os
import logging
import wave
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional
from contextlib import contextmanager

from slugify import slugify
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from fastapi import HTTPException

import yt_dlp
from app.config import BASE_DATA_DIR
from app.models.datasets import AudioDataset, DatasetStatus
from app.models.samples import SampleText, SampleStatus
from app.schemas.dataset import DatasetInitRequest
from app.services.segmentation_service import segment_audio
from app.notifications import notify_progress

logger = logging.getLogger(__name__)


class DatasetInitializationError(Exception):
    """Кастомное исключение для ошибок инициализации датасета"""
    pass


@contextmanager
def dataset_transaction(db: Session, dataset_id: int):
    """Context manager для транзакционной работы с датасетом"""
    dataset = None
    try:
        dataset = db.query(AudioDataset).filter(AudioDataset.id == dataset_id).first()
        if not dataset:
            raise DatasetInitializationError(f"Датасет с ID {dataset_id} не найден")
        
        yield dataset
        db.commit()
        
    except Exception as e:
        db.rollback()
        if dataset:
            dataset.status = DatasetStatus.ERROR
            dataset.last_update = datetime.utcnow()
            try:
                db.commit()
            except SQLAlchemyError:
                db.rollback()
        raise e


def cleanup_dataset_files(dataset: AudioDataset):
    """Очистка файлов датасета при ошибке"""
    try:
        # Удаляем исходный файл
        if dataset.source_rel_path:
            source_path = Path(BASE_DATA_DIR) / dataset.source_rel_path
            if source_path.exists():
                source_path.unlink()
                logger.info(f"Удален исходный файл: {source_path}")

        # Удаляем папку с сегментами
        if dataset.segments_rel_dir:
            segments_dir = Path(BASE_DATA_DIR) / dataset.segments_rel_dir
            if segments_dir.exists():
                shutil.rmtree(segments_dir)
                logger.info(f"Удалена папка с сегментами: {segments_dir}")
                
    except Exception as e:
        logger.error(f"Ошибка при cleanup файлов датасета {dataset.id}: {e}")


def create_dataset_entry(db: Session, url: str) -> int:
    """Создание записи датасета с транзакционной безопасностью"""
    try:
        name = "test"
        prefix = f"{name}_"
        existing_names = db.query(AudioDataset.name).filter(AudioDataset.name.like(f"{prefix}%")).all()
        existing_indices = [int(name[0].replace(prefix, '')) for name in existing_names if name[0].replace(prefix, '').isdigit()]
        next_index = max(existing_indices, default=0) + 1

        base_name = f"{name}_{next_index}"
        datasets_root = Path("datasets")
        source_rel_path = datasets_root / f"{base_name}.wav"
        segments_rel_dir = datasets_root / f"{base_name}_wavs"

        new_dataset = AudioDataset(
            name=base_name,
            url=url,
            source_rel_path=str(source_rel_path),
            segments_rel_dir=str(segments_rel_dir),
            status=DatasetStatus.INITIALIZING,
            created_at=datetime.utcnow(),
            last_update=datetime.utcnow()
        )

        db.add(new_dataset)
        db.commit()
        db.refresh(new_dataset)

        notify_progress(new_dataset.id, task="Создание записи в БД", progress=5)
        logger.info(f"Создан новый датасет ID={new_dataset.id}, name={base_name}")

        return new_dataset.id
        
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Ошибка создания записи датасета: {e}")
        raise DatasetInitializationError(f"Не удалось создать запись датасета: {e}")
    except Exception as e:
        db.rollback()
        logger.error(f"Неожиданная ошибка создания датасета: {e}")
        raise DatasetInitializationError(f"Неожиданная ошибка: {e}")


def initialize_dataset_service(dataset_id: int, data: DatasetInitRequest, db: Session):
    """Основная функция инициализации датасета с обработкой ошибок"""
    
    with dataset_transaction(db, dataset_id) as dataset:
        source_abs_path = Path(BASE_DATA_DIR) / dataset.source_rel_path
        segments_abs_dir = Path(BASE_DATA_DIR) / dataset.segments_rel_dir
        
        # Создаем необходимые директории
        try:
            source_abs_path.parent.mkdir(parents=True, exist_ok=True)
            segments_abs_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise DatasetInitializationError(f"Не удалось создать директории: {e}")

        # Этап 1: Скачивание аудио
        try:
            logger.info(f"Скачиваем аудио: {data.url}")
            download_audio_from_youtube(data.url, str(source_abs_path.with_suffix('')), dataset_id=dataset.id)
            notify_progress(dataset_id, task="Скачивание завершилось", progress=100)
            
            # Проверяем что файл действительно создался
            if not source_abs_path.exists():
                raise DatasetInitializationError("Файл не был создан после скачивания")
                
            logger.info("Изменяем статус: SAMPLING")
            dataset.status = DatasetStatus.SAMPLING
            dataset.last_update = datetime.utcnow()
            
        except Exception as e:
            logger.exception("Ошибка при скачивании аудио")
            cleanup_dataset_files(dataset)
            raise DatasetInitializationError(f"Ошибка при скачивании аудио: {e}")

        # Этап 2: Сегментация аудио
        try:
            logger.info("Сегментируем аудио")
            result = segment_audio(
                str(source_abs_path), 
                str(segments_abs_dir), 
                data.min_duration, 
                data.max_duration, 
                dataset_id=dataset.id
            )

            if result['status'] != 'success':
                raise DatasetInitializationError(result['message'])

            logger.info(f"Создано сегментов: {result['segments_count']}")
            logger.info(f"Статистика: {result['stats']}")

            # Получаем длительность исходного файла
            duration = get_audio_duration(str(source_abs_path))

            # Обновляем информацию о датасете
            dataset.count_of_samples = result['segments_count']
            dataset.duration = duration
            dataset.status = DatasetStatus.SAMPLED
            dataset.last_update = datetime.utcnow()

            # Создаем записи сэмплов
            create_sample_entries(db, dataset.id, str(segments_abs_dir))

            logger.info("Инициализация датасета завершена успешно")
            return {"status": "success", "dataset_id": dataset.id, "segments_count": result['segments_count']}

        except Exception as e:
            logger.exception("Ошибка при сегментации")
            cleanup_dataset_files(dataset)
            raise DatasetInitializationError(f"Ошибка при сегментации: {e}")


def download_audio_from_youtube(url: str, output_path: str, dataset_id: int, max_retries: int = 3) -> str:
    """Скачивание аудио с YouTube с retry логикой"""
    
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    def hook(d):
        if d['status'] == 'downloading':
            total = d.get('total_bytes') or d.get('total_bytes_estimate')
            downloaded = d.get('downloaded_bytes')
            if total and downloaded:
                percent = int(downloaded / total * 94) + 5
                notify_progress(dataset_id, task="Скачивание", progress=percent)

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True,
        'progress_hooks': [hook],
        'retries': max_retries,
        'socket_timeout': 30,
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    }

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            output_file = output_path + ".wav"
            if os.path.exists(output_file):
                logger.info(f"Успешно скачан файл: {output_file}")
                return output_file
            else:
                raise DatasetInitializationError("Файл не был создан после скачивания")
                
        except Exception as e:
            last_error = e
            logger.warning(f"Попытка {attempt + 1} не удалась: {e}")
            if attempt < max_retries:
                logger.info(f"Повтор через 5 секунд...")
                import time
                time.sleep(5)
            else:
                break

    raise DatasetInitializationError(f"Не удалось скачать после {max_retries + 1} попыток. Последняя ошибка: {last_error}")


def get_audio_duration(audio_path: str) -> float:
    """Получение длительности аудио файла с обработкой ошибок"""
    full_path = os.path.abspath(audio_path)
    if not os.path.exists(full_path):
        raise DatasetInitializationError(f"Аудио файл не найден: {full_path}")
    
    try:
        with wave.open(full_path, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate == 0:
                raise DatasetInitializationError("Некорректная частота дискретизации в аудио файле")
            return frames / float(rate)
    except wave.Error as e:
        raise DatasetInitializationError(f"Ошибка чтения WAV файла: {e}")
    except Exception as e:
        raise DatasetInitializationError(f"Неожиданная ошибка при получении длительности: {e}")


def create_sample_entries(db: Session, dataset_id: int, segments_abs_dir: str):
    """Создание записей сэмплов с обработкой ошибок"""
    if not os.path.exists(segments_abs_dir):
        raise DatasetInitializationError(f"Директория с сегментами не найдена: {segments_abs_dir}")
    
    try:
        logger.info(f"Создаём SampleText записи для {segments_abs_dir}")
        wav_files = sorted([f for f in os.listdir(segments_abs_dir) if f.endswith(".wav")])
        
        if not wav_files:
            raise DatasetInitializationError("Не найдено WAV файлов для создания сэмплов")

        samples_to_add = []
        for filename in wav_files:
            filepath = os.path.join(segments_abs_dir, filename)
            try:
                duration = get_audio_duration(filepath)
            except Exception as e:
                logger.warning(f"Не удалось получить длительность {filename}: {e}")
                duration = None

            sample = SampleText(
                dataset_id=dataset_id,
                filename=filename,
                text=None,
                duration=duration,
                status=SampleStatus.NEW,
                created_at=datetime.utcnow()
            )
            samples_to_add.append(sample)

        # Batch insert для лучшей производительности
        db.add_all(samples_to_add)
        db.commit()
        
        logger.info(f"Создано {len(wav_files)} сэмплов")
        
    except SQLAlchemyError as e:
        db.rollback()
        raise DatasetInitializationError(f"Ошибка создания записей сэмплов: {e}")
    except Exception as e:
        raise DatasetInitializationError(f"Неожиданная ошибка при создании сэмплов: {e}")