import os
import logging
import wave
from pathlib import Path
from datetime import datetime
from typing import Optional

from slugify import slugify
from sqlalchemy.orm import Session
from fastapi import HTTPException

import yt_dlp
from app.config import BASE_DATA_DIR
from app.models.datasets import AudioDataset, DatasetStatus
from app.models.samples import SampleText, SampleStatus
from app.schemas.dataset import DatasetInitRequest
from app.services.segmentation_service import segment_audio
from app.notifications import notify_progress

logger = logging.getLogger(__name__)


def create_dataset_entry(db: Session,  url: str) -> int:
    
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

    notify_progress(new_dataset.id, task="Создание записи в БД", progress=5)

    db.add(new_dataset)
    db.commit()
    db.refresh(new_dataset)

    logger.info(f"Создан новый датасет ID={new_dataset.id}, name={base_name}")

    return new_dataset.id


def initialize_dataset_service(dataset_id: int, data: DatasetInitRequest, db: Session):
    dataset: Optional[AudioDataset] = db.query(AudioDataset).filter(AudioDataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Датасет не найден")

    source_abs_path = Path(BASE_DATA_DIR) / dataset.source_rel_path
    segments_abs_dir = os.path.join(BASE_DATA_DIR, dataset.segments_rel_dir)
    os.makedirs(segments_abs_dir, exist_ok=True)
    
    


    try:
        logger.info(f"Скачиваем аудио: {data.url}")
        download_audio_from_youtube(data.url, str(source_abs_path.with_suffix('')), dataset_id=dataset.id)
        notify_progress(dataset_id, task="Скачивание завершилось", progress=100)
        logger.info("Изменяем статус: SAMPLING")
        dataset.status = DatasetStatus.SAMPLING
        db.commit()
    except Exception as e:
        logger.exception("Ошибка при скачивании аудио")
        dataset.status = DatasetStatus.ERROR
        db.commit()
        raise RuntimeError(f"Ошибка при скачивании аудио: {e}")
    
    try:
        

        logger.info("Сегментируем аудио")
        result = segment_audio(str(source_abs_path), str(segments_abs_dir), data.min_length, data.max_length, dataset_id=dataset.id)

        if result['status'] != 'success':
            raise RuntimeError(result['message'])

        logger.info(f"Создано сегментов: {result['segments_count']}")
        logger.info(f"Статистика: {result['stats']}")

        duration = get_audio_duration(str(source_abs_path))

        dataset.count_of_samples = result['segments_count']
        dataset.duration = duration
        dataset.status = DatasetStatus.SAMPLED
        dataset.last_update = datetime.utcnow()
        db.commit()

        create_sample_entries(db, dataset.id, segments_abs_dir)

        logger.info("Инициализация датасета завершена")

    except Exception as e:
        logger.exception("Ошибка при семплировании")
        dataset.status = DatasetStatus.ERROR
        db.commit()
        raise


def download_audio_from_youtube(url: str, output_path: str, dataset_id: int) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

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
        'progress_hooks': [hook]
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return output_path + ".wav"


def get_audio_duration(audio_path: str) -> float:
    full_path = os.path.abspath(audio_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Файл не найден: {full_path}")
    with wave.open(full_path, 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)


def create_sample_entries(db: Session, dataset_id: int, segments_abs_dir: str):
    logger.info(f"Создаём SampleText записи для {segments_abs_dir}")
    wav_files = sorted([f for f in os.listdir(segments_abs_dir) if f.endswith(".wav")])

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
        db.add(sample)

    db.commit()
    logger.info(f"Создано {len(wav_files)} сэмплов")
