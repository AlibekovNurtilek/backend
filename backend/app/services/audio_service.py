import os
from typing import List, Tuple

from fastapi import HTTPException
from sqlalchemy.orm import Session
from app.models.datasets import AudioDataset
from app.config import BASE_DATA_DIR
from fastapi.responses import FileResponse



def get_audio_filenames_by_dataset_id(
    dataset_id: int,
    db: Session,
    page: int = 1,
    limit: int = 10
) -> Tuple[List[str], int]:
    """
    Возвращает список имён .wav файлов по dataset_id с пагинацией.

    Args:
        dataset_id (int): ID датасета
        db (Session): SQLAlchemy сессия
        page (int): Номер страницы
        limit (int): Кол-во файлов на странице

    Returns:
        (List[str], int): Список имён файлов и общее количество файлов
    """
    # Получаем датасет
    dataset = db.query(AudioDataset).filter(AudioDataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Строим абсолютный путь к директории сегментов
    segments_dir = os.path.join(BASE_DATA_DIR, dataset.segments_rel_dir)

    if not os.path.isdir(segments_dir):
        raise HTTPException(status_code=404, detail=f"Segments directory not found")

    # Получаем все .wav файлы и сортируем по имени
    all_files = sorted([
        f for f in os.listdir(segments_dir)
        if f.endswith(".wav") and os.path.isfile(os.path.join(segments_dir, f))
    ])

    total = len(all_files)
    start = (page - 1) * limit
    end = start + limit
    page_files = all_files[start:end]

    return page_files, total


def get_audio_file_by_dataset_id_and_name(
    dataset_id: int,
    filename: str,
    db: Session
) -> FileResponse:
    """
    Возвращает аудиофайл по dataset_id и имени файла.

    Args:
        dataset_id (int): ID датасета
        filename (str): Имя файла (например, segment_001.wav)
        db (Session): SQLAlchemy сессия

    Returns:
        FileResponse: Стриминговый ответ
    """
    dataset = db.query(AudioDataset).filter(AudioDataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    segments_dir = os.path.join(BASE_DATA_DIR, dataset.segments_rel_dir)
    abs_file_path = os.path.abspath(os.path.join(segments_dir, filename))

    # 🔐 Безопасность: проверка что путь не выходит за пределы папки
    if not abs_file_path.startswith(os.path.abspath(segments_dir)):
        raise HTTPException(status_code=403, detail="Invalid filename or path traversal attempt")

    if not os.path.isfile(abs_file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(abs_file_path, media_type="audio/wav")
