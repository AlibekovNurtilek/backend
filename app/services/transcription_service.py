import os
import time
import logging
from datetime import datetime, timedelta

import requests
from fastapi import HTTPException
from sqlalchemy.orm import Session
from pydub import AudioSegment
from pydub.effects import normalize, low_pass_filter
from google.genai import types
from google import genai

from app.models.datasets import AudioDataset
from app.models.samples import SampleText
from app.models.data_status import SampleStatus, DatasetStatus
from app.config import BASE_DATA_DIR
from app.config import GEMINI_API_KEY
from app.tasks.notify_tasks import notify_progress_task




logger = logging.getLogger(__name__)

WHISPER_URL = "http://80.72.180.130:8330/transcribe"
WHISPER_HEADERS = {"X-API-Token": "togolokmoldo"}

def transcribe_dataset(dataset_id: int, transcriber_id: int, db: Session):
    dataset = db.query(AudioDataset).filter(AudioDataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Датасет не найден")

    if transcriber_id == 1:
        logger.info(f"Запускаем транскрипцию Whisper для dataset_id={dataset_id}")
        return transcribe_with_whisper(dataset, db)
    elif transcriber_id == 2:
        logger.info(f"Запускаем транскрипцию Gemini для dataset_id={dataset_id}")
        return transcribe_with_gemini(dataset, db)
    else:
        raise HTTPException(status_code=400, detail="Неверный transcriber_id. Ожидается 1 (Whisper) или 2 (Gemini)")




def transcribe_with_whisper(dataset: AudioDataset, db: Session):
    segments_dir = os.path.join(BASE_DATA_DIR, dataset.segments_rel_dir)
    if not os.path.exists(segments_dir):
        logger.error(f"Сегментированная директория не найдена: {segments_dir}")
        dataset.status = DatasetStatus.FAILED_TRANSCRIPTION
        db.commit()
        return {"message": "Сегментированная директория не найдена", "status": "error"}

    dataset.status = DatasetStatus.TRANSCRIBING
    db.commit()

    samples = db.query(SampleText).filter(
        SampleText.dataset_id == dataset.id,
        SampleText.status.in_([SampleStatus.NEW, SampleStatus.FAILED_TRANSCRIPTION])
    ).all()


    if not samples:
        logger.error(f"Нет сэмплов в базе для датасета ID={dataset.id}")
        dataset.status = DatasetStatus.FAILED_TRANSCRIPTION
        db.commit()
        return {"message": "Нет сэмплов для транскрипции", "status": "error"}

    success_count = 0
    fail_count = 0

    for i, sample in enumerate(samples, 1):
        file_path = os.path.join(segments_dir, sample.filename)

        if not os.path.exists(file_path):
            logger.warning(f"Файл не найден: {file_path}")
            sample.status = SampleStatus.FAILED_TRANSCRIPTION
            db.commit()
            fail_count += 1
            continue

        try:
            with open(file_path, "rb") as f:
                files = {"file": f}
                response = requests.post(WHISPER_URL, headers=WHISPER_HEADERS, files=files, timeout=30)

            if response.status_code == 200:
                json_data = response.json()
                text = json_data.get("text", "").strip()

                sample.text = text
                sample.status = SampleStatus.TRANSCRIBED
                db.commit()
                logger.info(f"Транскрипция успешна: {sample.filename}")
                success_count += 1

            else:
                logger.warning(f"Whisper не смог обработать {sample.filename}: {response.status_code}")
                sample.status = SampleStatus.FAILED_TRANSCRIPTION
                db.commit()
                fail_count += 1

        except Exception as e:
            logger.exception(f"Ошибка при транскрипции {sample.filename}: {e}")
            sample.status = SampleStatus.FAILED_TRANSCRIPTION
            db.commit()
            fail_count += 1

        progress = int(i / len(samples) * 100)
        notify_progress_task.delay(
            dataset_id=dataset.id,
            task="Транскрипция (Whisper)",
            progress=progress
        )

    # Обновление статуса датасета
    if success_count == 0:
        dataset.status = DatasetStatus.FAILED_TRANSCRIPTION
    elif fail_count > 0:
        dataset.status = DatasetStatus.SEMY_TRANSCRIBED  # Частичная
    else:
        dataset.status = DatasetStatus.REVIEW  # Всё успешно

    db.commit()
    return {
        "message": "Транскрипция завершена",
        "success": success_count,
        "failed": fail_count,
        "status": dataset.status
    }

# Настройка Gemini
GOOGLE_API_KEY = GEMINI_API_KEY
client = genai.Client(api_key=GOOGLE_API_KEY)

PROMPT = """
Transcribe the provided audio into Kyrgyz text with maximum accuracy. 
"""  

REQUESTS_PER_MINUTE = 10
REQUESTS_PER_DAY = 500
request_count_minute = 0
request_count_day = 0
last_minute = datetime.now()
last_day = datetime.now()

GENERATION_CONFIG = types.GenerateContentConfig(
    temperature=0.8,      # 0.0–0.3 — обычно оптимально для транскрипции
)



def check_gemini_limits():
    global request_count_minute, request_count_day, last_minute, last_day
    now = datetime.now()

    if now - last_minute >= timedelta(minutes=1):
        request_count_minute = 0
        last_minute = now
    if now - last_day >= timedelta(days=1):
        request_count_day = 0
        last_day = now

    if request_count_day >= REQUESTS_PER_DAY:
        return False, "дневной лимит"
    if request_count_minute >= REQUESTS_PER_MINUTE:
        return False, "минутный лимит"
    return True, None


def preprocess_audio(audio_path, speed_factor=0.9):
    try:
        audio = AudioSegment.from_file(audio_path)
        audio = normalize(audio)
        audio = low_pass_filter(audio, 3000)
        audio = audio.speedup(playback_speed=1 / speed_factor, crossfade=0)
        temp_path = os.path.join(os.path.dirname(audio_path), f"temp_{os.getpid()}.mp3")
        audio.export(temp_path, format="mp3")
        return temp_path
    except Exception as e:
        logger.warning(f"[Gemini] Ошибка при предобработке {audio_path}: {e}")
        return audio_path


def transcribe_file_with_gemini(audio_path: str):
    global request_count_minute, request_count_day

    can_proceed, reason = check_gemini_limits()
    if not can_proceed:
        return False, None, reason

    if not os.path.exists(audio_path):
        return False, None, "Файл не найден"

    processed_path = preprocess_audio(audio_path)

    try:
        uploaded_file = client.files.upload(file=processed_path)
        response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[PROMPT, uploaded_file],
        config=GENERATION_CONFIG,
        )

        request_count_minute += 1
        request_count_day += 1

        return True, response.text.strip(), None

    except Exception as e:
        return False, None, str(e)

    finally:
        if processed_path != audio_path and os.path.exists(processed_path):
            try:
                os.remove(processed_path)
            except:
                pass


def transcribe_with_gemini(dataset: AudioDataset, db: Session):
    dataset.status = DatasetStatus.TRANSCRIBING
    db.commit()

    samples = db.query(SampleText).filter(
        SampleText.dataset_id == dataset.id,
        SampleText.status.in_([SampleStatus.NEW, SampleStatus.FAILED_TRANSCRIPTION])
    ).all()
    
    if not samples:
        dataset.status = DatasetStatus.FAILED_TRANSCRIPTION
        db.commit()
        return {"message": "Нет сэмплов"}

    success_count = 0
    failed_count = 0
    base_dir = os.path.join(BASE_DATA_DIR, dataset.segments_rel_dir)

    for i, sample in enumerate(samples, 1):
        path = os.path.join(base_dir, sample.filename)
        if not os.path.exists(path):
            sample.status = SampleStatus.FAILED_TRANSCRIPTION
            db.commit()
            failed_count += 1
            continue

        success, result, error = transcribe_file_with_gemini(path)

        if not success and "лимит" in (error or "").lower():
            logger.warning("[Gemini] Лимит — ждём 90 сек и повтор...")
            time.sleep(90)
            success, result, error = transcribe_file_with_gemini(path)

        if success:
            sample.text = result
            sample.status = SampleStatus.TRANSCRIBED
            logger.info(f"[Gemini] Успешно: {sample.filename}")
            success_count += 1
        else:
            sample.status = SampleStatus.FAILED_TRANSCRIPTION
            logger.warning(f"[Gemini] Ошибка для {sample.filename}: {error}")
            failed_count += 1

        db.commit()
        
        progress = int(i / len(samples) * 100)
        notify_progress_task.delay(
            dataset_id=dataset.id,
            task="Транскрипция (Gemini)",
            progress=progress
        )


    if success_count == 0:
        dataset.status = DatasetStatus.FAILED_TRANSCRIPTION
    elif failed_count > 0:
        dataset.status = DatasetStatus.SEMY_TRANSCRIBED
    else:
        dataset.status = DatasetStatus.REVIEW

    db.commit()
    return {
        "dataset_id": dataset.id,
        "transcribed": success_count,
        "failed": failed_count,
        "total": len(samples),
        "status": dataset.status
    }