from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Annotated, Optional
from datetime import datetime
from fastapi import Query
from datetime import datetime
from app.db import SessionLocal
from pydantic import BaseModel
from app.models.datasets import AudioDataset
from app.services import dataset_service
from app.schemas.dataset import DatasetListResponse
from app.schemas.dataset import (
    DatasetCreate,
    DatasetUpdate,
    DatasetOut,
    DatasetInitRequest,
    DatasetImageUpdate
)
from app.tasks.initialize_dataset_tasks import initialize_dataset_task
from app.services.initialize_service import create_dataset_entry
import logging
logger = logging.getLogger(__name__)



router = APIRouter(prefix="/datasets", tags=["datasets"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



# ---------------------
# 📍 Роуты
# ---------------------




@router.get("/", response_model=DatasetListResponse)
def get_all_datasets(
    limit: int = Query(10, ge=1),
    offset: int = Query(0, ge=0),
    status: Optional[str] = None,
    name_search: Optional[str] = None,
    created_from: Optional[datetime] = None,
    created_to: Optional[datetime] = None,
    db: Session = Depends(get_db),
):
    return dataset_service.get_all_datasets(
        db=db,
        limit=limit,
        offset=offset,
        status=status,
        name_search=name_search,
        created_from=created_from,
        created_to=created_to,
    )

@router.get("/{dataset_id}", response_model=DatasetOut)
def get_dataset_by_id(dataset_id: int, db: Session = Depends(get_db)):
    return dataset_service.get_dataset_by_id(dataset_id, db)


@router.put("/{dataset_id}", response_model=DatasetOut)
def update_dataset(dataset_id: int, dataset: DatasetUpdate, db: Session = Depends(get_db)):
    return dataset_service.update_dataset(dataset_id, dataset, db)

@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(dataset_id: int, db: Session = Depends(get_db)):
    dataset_service.delete_dataset(dataset_id, db)
    return


@router.post("/initialize")
def initialize_dataset(data: DatasetInitRequest, db: Session = Depends(get_db)):
    # Шаг 0: проверяем наличие такого же URL в базе
    # existing_dataset = db.query(AudioDataset).filter(AudioDataset.url == data.url).first()
    # if existing_dataset:
    #     logger.info(f"Датасет с таким URL уже существует: ID={existing_dataset.id}, name={existing_dataset.name}")
    #     return {
    #         "message": "Видео уже загружено",
    #         "dataset_id": existing_dataset.id,
    #         "status": existing_dataset.status,
    #     }

    # Шаг 1: создаём запись в БД со статусом INITIALIZING
    dataset_id = create_dataset_entry(db, data.url)

    # Шаг 2: запускаем фоновую задачу
    task = initialize_dataset_task.delay(dataset_id, data.dict())

    return {
        "message": "Задача инициализации добавлена",
        "task_id": task.id,
        "dataset_id": dataset_id
    }

