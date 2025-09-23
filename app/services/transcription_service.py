import os
import time
import logging
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, Any
from collections import defaultdict

import requests
from fastapi import HTTPException
from sqlalchemy.orm import Session
from pydub import AudioSegment
from pydub.effects import normalize, low_pass_filter
from google.genai import types
from google import genai
from requests.exceptions import RequestException, Timeout, ConnectionError

from app.models.datasets import AudioDataset
from app.models.samples import SampleText
from app.models.data_status import SampleStatus, DatasetStatus
from app.config import BASE_DATA_DIR, GEMINI_API_KEY
from app.tasks.notify_tasks import notify_progress_task

logger = logging.getLogger(__name__)

# Whisper настройки
WHISPER_URL = "http://80.72.180.130:8330/transcribe"
WHISPER_HEADERS = {"X-API-Token": "togolokmoldo"}

# Gemini настройки
client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_PROMPT = """
Transcribe the provided audio into Kyrgyz text with maximum accuracy. Follow these guidelines:

1. Listen carefully and transcribe what you hear in Kyrgyz language.
2. Return only the Kyrgyz text without any additional comments or explanations.
3. Use your best judgment to transcribe even if some words are not perfectly clear.
4. Add appropriate punctuation (periods, commas, question marks) based on the speech rhythm and intonation.
5. If you hear Kyrgyz speech, transcribe it as accurately as possible, even if there's some background noise.
6. Only return "not valid" if the audio is completely silent, contains no speech, or is entirely in a different language.
7. For unclear words, make your best guess based on context and common Kyrgyz vocabulary.

Goal: Provide the most accurate Kyrgyz transcription possible, focusing on capturing the actual speech content.
"""

# Лимиты для разных моделей Gemini
MODEL_LIMITS = {
    2: {"rpm": 15, "rpd": 1500, "name": "gemini-2.0-flash"},
    3: {"rpm": 10, "rpd": 1000, "name": "gemini-2.5-flash"}
}

GENERATION_CONFIG = types.GenerateContentConfig(temperature=0.8)

# Глобальные счетчики запросов по моделям
request_trackers = {
    model_id: {
        "minute_requests": [],
        "day_requests": [],
        "last_request": None
    }
    for model_id in MODEL_LIMITS.keys()
}


class TranscriptionError(Exception):
    """Базовый класс для ошибок транскрипции"""
    pass


class RateLimitError(TranscriptionError):
    """Ошибка превышения лимитов"""
    pass


class AudioProcessingError(TranscriptionError):
    """Ошибка обработки аудио"""
    pass


def transcribe_dataset(dataset_id: int, transcriber_id: int, db: Session) -> Dict[str, Any]:
    """Главная функция транскрипции датасета"""
    dataset = db.query(AudioDataset).filter(AudioDataset.id == dataset_id).first()
    if not dataset:
        logger.error(f"Датасет не найден: {dataset_id}")
        raise HTTPException(status_code=404, detail="Датасет не найден")

    logger.info(f"Запуск транскрипции dataset_id={dataset_id}, transcriber_id={transcriber_id}")
    
    try:
        if transcriber_id == 1:
            return transcribe_with_whisper(dataset, db)
        elif transcriber_id in [2, 3]:
            return transcribe_with_gemini(dataset, db, model_id=transcriber_id)
        else:
            raise HTTPException(status_code=400, detail="Неверный transcriber_id")
    except Exception as e:
        logger.error(f"Критическая ошибка транскрипции dataset_id={dataset_id}: {str(e)}", exc_info=True)
        dataset.status = DatasetStatus.FAILED_TRANSCRIPTION
        db.commit()
        raise


def transcribe_with_whisper(dataset: AudioDataset, db: Session) -> Dict[str, Any]:
    """Транскрипция через Whisper API"""
    segments_dir = os.path.join(BASE_DATA_DIR, dataset.segments_rel_dir)
    
    if not os.path.exists(segments_dir):
        logger.error(f"Сегментированная директория не найдена: {segments_dir}")
        dataset.status = DatasetStatus.FAILED_TRANSCRIPTION
        db.commit()
        return {"message": "Сегментированная директория не найдена", "status": "error"}

    dataset.status = DatasetStatus.TRANSCRIBING
    db.commit()

    samples = _get_samples_for_transcription(dataset.id, db)
    if not samples:
        logger.error(f"Нет сэмплов для транскрипции dataset_id={dataset.id}")
        dataset.status = DatasetStatus.FAILED_TRANSCRIPTION
        db.commit()
        return {"message": "Нет сэмплов для транскрипции", "status": "error"}

    success_count = 0
    fail_count = 0

    for i, sample in enumerate(samples, 1):
        file_path = os.path.join(segments_dir, sample.filename)
        
        if not os.path.exists(file_path):
            logger.warning(f"Файл не найден: {file_path}")
            _update_sample_status(sample, SampleStatus.FAILED_TRANSCRIPTION, db)
            fail_count += 1
            continue

        try:
            transcription_result = _transcribe_with_whisper_api(file_path, sample.filename)
            
            if transcription_result["success"]:
                sample.text = transcription_result["text"]
                _update_sample_status(sample, SampleStatus.TRANSCRIBED, db)
                success_count += 1
                logger.info(f"Whisper успешно обработал: {sample.filename}")
            else:
                _update_sample_status(sample, SampleStatus.FAILED_TRANSCRIPTION, db)
                fail_count += 1
                logger.warning(f"Whisper не смог обработать {sample.filename}: {transcription_result['error']}")

        except Exception as e:
            logger.error(f"Ошибка при транскрипции Whisper {sample.filename}: {str(e)}", exc_info=True)
            _update_sample_status(sample, SampleStatus.FAILED_TRANSCRIPTION, db)
            fail_count += 1

        _send_progress_notification(dataset.id, "Транскрипция (Whisper)", i, len(samples))

    _update_dataset_status(dataset, success_count, fail_count, db)
    
    return {
        "message": "Транскрипция завершена",
        "success": success_count,
        "failed": fail_count,
        "status": dataset.status
    }


def _transcribe_with_whisper_api(file_path: str, filename: str) -> Dict[str, Any]:
    """Отправка файла в Whisper API"""
    try:
        with open(file_path, "rb") as f:
            files = {"file": f}
            response = requests.post(
                WHISPER_URL, 
                headers=WHISPER_HEADERS, 
                files=files, 
                timeout=60
            )

        if response.status_code == 200:
            json_data = response.json()
            text = json_data.get("text", "").strip()
            return {"success": True, "text": text}
        else:
            return {
                "success": False, 
                "error": f"HTTP {response.status_code}: {response.text}"
            }

    except Timeout:
        return {"success": False, "error": "Timeout при запросе к Whisper API"}
    except ConnectionError:
        return {"success": False, "error": "Ошибка соединения с Whisper API"}
    except RequestException as e:
        return {"success": False, "error": f"Ошибка запроса: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": f"Неожиданная ошибка: {str(e)}"}


def transcribe_with_gemini(dataset: AudioDataset, db: Session, model_id: int) -> Dict[str, Any]:
    """Транскрипция через Gemini API"""
    dataset.status = DatasetStatus.TRANSCRIBING
    db.commit()

    samples = _get_samples_for_transcription(dataset.id, db)
    if not samples:
        dataset.status = DatasetStatus.FAILED_TRANSCRIPTION
        db.commit()
        return {"message": "Нет сэмплов для транскрипции"}

    success_count = 0
    failed_count = 0
    base_dir = os.path.join(BASE_DATA_DIR, dataset.segments_rel_dir)
    model_name = MODEL_LIMITS[model_id]["name"]

    logger.info(f"Начинаем транскрипцию Gemini с моделью {model_name}, файлов: {len(samples)}")

    for i, sample in enumerate(samples, 1):
        file_path = os.path.join(base_dir, sample.filename)
        
        if not os.path.exists(file_path):
            logger.warning(f"Файл не найден: {file_path}")
            _update_sample_status(sample, SampleStatus.FAILED_TRANSCRIPTION, db)
            failed_count += 1
            continue

        # Попытки транскрипции с обработкой лимитов
        max_attempts = 3
        attempt = 1
        
        while attempt <= max_attempts:
            try:
                success, result, error = _transcribe_file_with_gemini(file_path, model_id, sample.filename)
                
                if success:
                    sample.text = result
                    _update_sample_status(sample, SampleStatus.TRANSCRIBED, db)
                    success_count += 1
                    logger.info(f"Gemini успешно обработал: {sample.filename} (попытка {attempt})")
                    break
                
                elif "rate limit" in error.lower() or "429" in error:
                    wait_time = _calculate_backoff_time(attempt)
                    logger.warning(f"Rate limit для {sample.filename}, ждем {wait_time}сек (попытка {attempt}/{max_attempts})")
                    time.sleep(wait_time)
                    attempt += 1
                    
                else:
                    # Не rate limit ошибка - не повторяем
                    logger.error(f"Gemini ошибка для {sample.filename}: {error}")
                    _update_sample_status(sample, SampleStatus.FAILED_TRANSCRIPTION, db)
                    failed_count += 1
                    break
                    
            except Exception as e:
                logger.error(f"Неожиданная ошибка при обработке {sample.filename}: {str(e)}", exc_info=True)
                if attempt == max_attempts:
                    _update_sample_status(sample, SampleStatus.FAILED_TRANSCRIPTION, db)
                    failed_count += 1
                    break
                attempt += 1
                time.sleep(_calculate_backoff_time(attempt))

        if attempt > max_attempts and sample.status != SampleStatus.TRANSCRIBED:
            logger.error(f"Исчерпаны попытки для {sample.filename}")
            _update_sample_status(sample, SampleStatus.FAILED_TRANSCRIPTION, db)
            failed_count += 1

        _send_progress_notification(dataset.id, f"Транскрипция ({model_name})", i, len(samples))

    _update_dataset_status(dataset, success_count, failed_count, db)
    
    logger.info(f"Завершена транскрипция Gemini: успешно={success_count}, ошибок={failed_count}")
    
    return {
        "dataset_id": dataset.id,
        "transcribed": success_count,
        "failed": failed_count,
        "total": len(samples),
        "status": dataset.status,
        "model": model_name
    }


def _transcribe_file_with_gemini(file_path: str, model_id: int, filename: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """Транскрипция одного файла через Gemini API"""
    
    # Проверка лимитов перед запросом
    can_proceed, reason = _check_gemini_limits(model_id)
    if not can_proceed:
        return False, None, f"Rate limit: {reason}"

    # Предобработка аудио
    processed_path = None
    try:
        processed_path = _preprocess_audio(file_path)
        
        # Загрузка файла в Gemini
        uploaded_file = client.files.upload(file=processed_path)
        
        # Генерация транскрипции
        response = client.models.generate_content(
            model=MODEL_LIMITS[model_id]["name"],
            contents=[GEMINI_PROMPT, uploaded_file],
            config=GENERATION_CONFIG,
        )
        
        # Обновление счетчиков
        _update_request_tracker(model_id)
        
        if response.text and response.text.strip():
            return True, response.text.strip(), None
        else:
            return False, None, "Пустой ответ от Gemini API"

    except Exception as e:
        error_msg = str(e)
        
        # Определение типа ошибки
        if "429" in error_msg or "rate limit" in error_msg.lower():
            return False, None, f"Rate limit error: {error_msg}"
        elif "400" in error_msg:
            return False, None, f"Bad request: {error_msg}"
        elif "401" in error_msg or "403" in error_msg:
            return False, None, f"Auth error: {error_msg}"
        elif "500" in error_msg or "503" in error_msg:
            return False, None, f"Server error: {error_msg}"
        else:
            logger.error(f"Неизвестная ошибка Gemini API для {filename}: {error_msg}", exc_info=True)
            return False, None, f"Unknown error: {error_msg}"
            
    finally:
        # Очистка временного файла
        if processed_path and processed_path != file_path and os.path.exists(processed_path):
            try:
                os.remove(processed_path)
            except Exception as e:
                logger.warning(f"Не удалось удалить временный файл {processed_path}: {str(e)}")


def _check_gemini_limits(model_id: int) -> Tuple[bool, Optional[str]]:
    """Проверка лимитов для конкретной модели Gemini"""
    now = datetime.now()
    limits = MODEL_LIMITS[model_id]
    tracker = request_trackers[model_id]
    
    # Очистка старых запросов (старше минуты)
    tracker["minute_requests"] = [
        req_time for req_time in tracker["minute_requests"]
        if now - req_time < timedelta(minutes=1)
    ]
    
    # Очистка старых запросов (старше дня)
    tracker["day_requests"] = [
        req_time for req_time in tracker["day_requests"]
        if now - req_time < timedelta(days=1)
    ]
    
    # Проверка минутного лимита
    if len(tracker["minute_requests"]) >= limits["rpm"]:
        return False, f"минутный лимит ({limits['rpm']} RPM)"
    
    # Проверка дневного лимита
    if len(tracker["day_requests"]) >= limits["rpd"]:
        return False, f"дневной лимит ({limits['rpd']} RPD)"
    
    return True, None


def _update_request_tracker(model_id: int) -> None:
    """Обновление счетчика запросов"""
    now = datetime.now()
    tracker = request_trackers[model_id]
    
    tracker["minute_requests"].append(now)
    tracker["day_requests"].append(now)
    tracker["last_request"] = now


def _preprocess_audio(audio_path: str) -> str:
    """Предобработка аудиофайла"""
    try:
        audio = AudioSegment.from_file(audio_path)
        audio = normalize(audio)
        audio = low_pass_filter(audio, 3000)
        audio = audio.speedup(playback_speed=1.11, crossfade=0)  # Немного замедляем
        
        temp_path = os.path.join(os.path.dirname(audio_path), f"temp_{os.getpid()}_{int(time.time())}.mp3")
        audio.export(temp_path, format="mp3")
        
        logger.debug(f"Аудио предобработано: {audio_path} -> {temp_path}")
        return temp_path
        
    except Exception as e:
        logger.warning(f"Ошибка предобработки {audio_path}: {str(e)}")
        return audio_path


def _calculate_backoff_time(attempt: int) -> int:
    """Вычисление времени ожидания с экспоненциальным backoff"""
    base_wait = 60  # базовое время ожидания в секундах
    return min(base_wait * (2 ** (attempt - 1)), 300)  # максимум 5 минут


def _get_samples_for_transcription(dataset_id: int, db: Session):
    """Получение сэмплов для транскрипции"""
    return db.query(SampleText).filter(
        SampleText.dataset_id == dataset_id,
        SampleText.status.in_([SampleStatus.NEW, SampleStatus.FAILED_TRANSCRIPTION])
    ).all()


def _update_sample_status(sample: SampleText, status: SampleStatus, db: Session) -> None:
    """Обновление статуса сэмпла"""
    sample.status = status
    db.commit()


def _update_dataset_status(dataset: AudioDataset, success_count: int, fail_count: int, db: Session) -> None:
    """Обновление статуса датасета на основе результатов"""
    if success_count == 0:
        dataset.status = DatasetStatus.FAILED_TRANSCRIPTION
    elif fail_count > 0:
        dataset.status = DatasetStatus.SEMY_TRANSCRIBED
    else:
        dataset.status = DatasetStatus.REVIEW
    
    db.commit()
    logger.info(f"Статус датасета {dataset.id} обновлен на {dataset.status}")


def _send_progress_notification(dataset_id: int, task_name: str, current: int, total: int) -> None:
    """Отправка уведомления о прогрессе"""
    progress = int(current / total * 100)
    try:
        notify_progress_task.delay(
            dataset_id=dataset_id,
            task=task_name,
            progress=progress
        )
    except Exception as e:
        logger.warning(f"Ошибка отправки уведомления о прогрессе: {str(e)}")