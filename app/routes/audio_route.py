from fastapi import APIRouter, Depends, Request, Query, HTTPException
from sqlalchemy.orm import Session
from app.db import SessionLocal
from app.services.audio_service import (
    get_audio_filenames_by_dataset_id,
    get_audio_file_by_dataset_id_and_name
)
from fastapi.responses import FileResponse
from app.auth.utils import get_current_user


router = APIRouter(prefix="/audio", tags=["audio"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/list")
def list_audio_segments(
    dataset_id: int = Query(..., description="ID датасета"),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1),
    db: Session = Depends(get_db),
    user=Depends(get_current_user)
):
    files, total = get_audio_filenames_by_dataset_id(dataset_id, db, page, limit)
    return {
        "dataset_id": dataset_id,
        "page": page,
        "limit": limit,
        "total": total,
        "files": files
    }

@router.get("/stream")
def stream_audio_segment(
    dataset_id: int = Query(..., description="ID датасета"),
    filename: str = Query(..., description="Имя сегмента (например: segment_001.wav)"),
    db: Session = Depends(get_db),
    user=Depends(get_current_user)
):
    return get_audio_file_by_dataset_id_and_name(dataset_id, filename, db)

