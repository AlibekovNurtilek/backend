from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db import SessionLocal
from app.tasks.transcription_tasks import transcribe_dataset_task
from pydantic import BaseModel, validator

router = APIRouter(prefix="/transcribe", tags=["Transcription"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class TranscriptionRequest(BaseModel):
    model_name: str  # теперь обязательное поле

    @validator("model_name")
    def validate_model_name(cls, v):
        allowed_models = ["Aitil Whisper", "Gemini 2.0", "Gemini 2.5"]
        if v not in allowed_models:
            raise ValueError(f"model_name должен быть одним из {allowed_models}")
        return v

@router.post("/{dataset_id}")
def start_transcription(dataset_id: int, request: TranscriptionRequest, db: Session = Depends(get_db)):
    # Можно сопоставить model_name с transcriber_id, если нужно
    model_to_id = {
        "Aitil Whisper": 1,
        "Gemini 2.0": 2,
        "Gemini 2.5": 3,
    }
    transcriber_id = model_to_id.get(request.model_name)

    task = transcribe_dataset_task.delay(dataset_id, transcriber_id)
    
    return {
        "message": "Транскрипция запущена",
        "task_id": task.id,
        "dataset_id": dataset_id,
        "model_name": request.model_name,
        "transcriber_id": transcriber_id
    }
