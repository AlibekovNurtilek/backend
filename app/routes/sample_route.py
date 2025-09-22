from fastapi import APIRouter, Depends, status, Query
from sqlalchemy.orm import Session
from typing import List
from app.db import SessionLocal
from app.schemas.sample import (
    SampleCreate, SampleUpdate, SampleOut, DatasetSamplesResponse
)
from app.services import sample_service
from app.models.data_status import SampleStatus

router = APIRouter(prefix="/samples", tags=["samples"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/", response_model=List[SampleOut])
def get_all_samples(db: Session = Depends(get_db)):
    return sample_service.get_all_samples(db)


@router.get("/{sample_id}", response_model=SampleOut)
def get_sample_by_id(sample_id: int, db: Session = Depends(get_db)):
    return sample_service.get_sample_by_id(sample_id, db)

@router.get("/by-dataset/{dataset_id}", response_model=DatasetSamplesResponse)
def get_samples_by_dataset_id(
    dataset_id: int,
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1),
    status: SampleStatus | None = Query(None, description="Фильтрация по статусу"),
    search: str | None = Query(None, description="Поиск по filename или text"),
    from_index: int | None = Query(None, ge=0, description="Начальный индекс в отсортированном списке по filename"),
    to_index: int | None = Query(None, ge=0, description="Конечный индекс в отсортированном списке по filename"),
    db: Session = Depends(get_db)
):
    return sample_service.get_samples_by_dataset_id(
        dataset_id=dataset_id,
        db=db,
        page=page,
        limit=limit,
        status=status,
        search=search,
        from_index=from_index,
        to_index=to_index
    )


@router.post("/", response_model=SampleOut, status_code=status.HTTP_201_CREATED)
def create_sample(sample: SampleCreate, db: Session = Depends(get_db)):
    return sample_service.create_sample(sample, db)


@router.put("/{sample_id}", response_model=SampleOut)
def update_sample(sample_id: int, sample: SampleUpdate, db: Session = Depends(get_db)):
    return sample_service.update_sample(sample_id, sample, db)


@router.delete("/{sample_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_sample(sample_id: int, db: Session = Depends(get_db)):
    sample_service.delete_sample(sample_id, db)
    return

@router.post("/{sample_id}/approve", response_model=SampleOut)
def approve_sample_route(sample_id: int, db: Session = Depends(get_db)):
    return sample_service.approve_sample(sample_id, db)

@router.post("/{sample_id}/reject", response_model=SampleOut)
def reject_sample_route(sample_id: int, db: Session = Depends(get_db)):
    return sample_service.reject_sample(sample_id, db)