from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db import SessionLocal
from app.tasks.transcription_tasks import transcribe_dataset_task
from pydantic import BaseModel

router = APIRouter(prefix="/transcribe", tags=["Transcription"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class TranscriptionRequest(BaseModel):
    transcriber_id: int

@router.post("/{dataset_id}")
def start_transcription(dataset_id: int, request: TranscriptionRequest, db: Session = Depends(get_db)):
    if request.transcriber_id not in (1, 2):
        raise HTTPException(status_code=400, detail="transcriber_id должен быть 1 (Whisper) или 2 (Gemini)")

    task = transcribe_dataset_task.delay(dataset_id, request.transcriber_id)
    
    return {
        "message": "Транскрипция запущена",
        "task_id": task.id,
        "dataset_id": dataset_id,
        "transcriber_id": request.transcriber_id
    }