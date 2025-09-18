import os
import numpy as np
import librosa
import soundfile as sf
import logging
import torch
from app.tasks.notify_tasks import notify_progress_task

logger = logging.getLogger(__name__)

# === Загрузка Silero VAD (один раз при старте) ===
def get_silero_vad_model():
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False
    )
    (get_speech_timestamps, _, _, _, _) = utils
    return get_speech_timestamps, model

_silero_cache = None

def segment_audio(
    input_wav_path,
    output_dir,
    min_length=1.5,
    max_length=12.0,
    min_silence_duration=0.3,
    speech_pad=0.05,
    frame_length=512,
    hop_length=160,
    max_extension=1.5,
    allow_short_final=True,
    debug=False,
    dataset_id=1
):
    global _silero_cache

    # === Шаг 1: Загрузка WAV ===
    notify_progress_task.delay(dataset_id=dataset_id, task="Загрузка WAV", progress=0)
    try:
        y, sr = librosa.load(input_wav_path, sr=16000, mono=True)
    except Exception as e:
        return {"status": "error", "message": f"Failed to load audio: {str(e)}"}

    notify_progress_task.delay(dataset_id=dataset_id, task="Загрузка WAV", progress=10)

    if sr != 16000:
        return {"status": "error", "message": "Audio must be 16kHz."}

    total_duration = len(y) / sr

    # === Шаг 2: VAD ===
    notify_progress_task.delay(dataset_id=dataset_id, task="VAD анализ", progress=15)
    if _silero_cache is None:
        try:
            _silero_cache = get_silero_vad_model()
        except Exception as e:
            return {"status": "error", "message": f"Failed to load Silero VAD: {str(e)}"}
    get_speech_timestamps, model = _silero_cache
    notify_progress_task.delay(dataset_id=dataset_id, task="VAD анализ", progress=20)

    # === Шаг 3: Подготовка аудио ===
    notify_progress_task.delay(dataset_id=dataset_id, task="Подготовка аудио", progress=25)
    wav_tensor = torch.from_numpy(y).float().unsqueeze(0)
    notify_progress_task.delay(dataset_id=dataset_id, task="Подготовка аудио", progress=30)

    # === Шаг 4: Получение сегментов речи ===
    try:
        speech_segments = get_speech_timestamps(
            wav_tensor,
            model,
            threshold=0.5,
            min_silence_duration_ms=int(min_silence_duration * 1000),
            speech_pad_ms=int(speech_pad * 1000)
        )
    except Exception as e:
        return {"status": "error", "message": f"VAD processing failed: {str(e)}"}

    active_intervals = [
        {"start": s["start"] / 16000.0, "end": s["end"] / 16000.0}
        for s in speech_segments
    ]

    if not active_intervals:
        return {"status": "warning", "message": "No speech detected", "segments_count": 0}

    # === Шаг 5: Сегментация речи (30–50%) ===
    total_seg = len(active_intervals)
    merged = []
    current = active_intervals[0]
    for idx, next_seg in enumerate(active_intervals[1:], 1):
        if next_seg["start"] - current["end"] < min_silence_duration:
            current["end"] = next_seg["end"]
        else:
            merged.append(current)
            current = next_seg

        progress = 30 + int((idx / total_seg) * 20)
        notify_progress_task.delay(dataset_id=dataset_id, task="Сегментация речи", progress=progress)

    merged.append(current)

    # === Шаг 6: Слияние интервалов (50–60%) ===
    total_merge = len(merged)

    os.makedirs(output_dir, exist_ok=True)
    chunks = []
    durations = []
    saved = 0
    current_time = 0.0
    split_points = []
    estimated_segments = total_duration / max_length

    # === Шаг 7: Формирование сегментов (60–90%) ===
    while current_time < total_duration:
        start_time = current_time
        min_end_time = start_time + min_length
        max_end_time = start_time + max_length
        hard_end_time = min(max_end_time, total_duration)
        search_end_time = min(max_end_time + max_extension, total_duration)

        cut_time = None
        for interval in merged:
            if interval["start"] > min_end_time and interval["start"] <= search_end_time:
                prev_end = current_time
                for prev in merged:
                    if prev["end"] < interval["start"]:
                        prev_end = max(prev_end, prev["end"])
                if interval["start"] - prev_end >= min_silence_duration:
                    cut_time = interval["start"]
                    break

        if cut_time is None:
            cut_time = hard_end_time

        seg_duration = cut_time - start_time
        if seg_duration < min_length:
            if cut_time >= total_duration and allow_short_final:
                cut_time = total_duration
            else:
                cut_time = min(start_time + max_length, total_duration)
                if cut_time - start_time < min_length:
                    if allow_short_final and start_time < total_duration:
                        cut_time = total_duration
                    else:
                        break

        if cut_time <= start_time + 1e-3:
            cut_time = min(start_time + 0.2, total_duration)
            if cut_time <= start_time:
                break

        start_sample = int(start_time * sr)
        end_sample = int(cut_time * sr)
        segment = y[start_sample:end_sample]

        if len(segment) == 0:
            break

        filename = f"segment_{saved + 1:04d}.wav"
        filepath = os.path.join(output_dir, filename)
        sf.write(filepath, segment, sr, subtype='PCM_16')
        logger.info(f"file: {filepath} saved")

        chunks.append({"start": start_time, "end": cut_time, "samples": segment})
        durations.append(cut_time - start_time)
        split_points.append(cut_time)
        saved += 1

        progress = 60 + int((saved / estimated_segments) * 30)
        progress = min(progress, 90)
        notify_progress_task.delay(dataset_id=dataset_id, task="Формирование сегментов", progress=progress)

        current_time = cut_time
        if current_time >= total_duration - 1e-3:
            break

    # === Шаг 8: Финал ===
    notify_progress_task.delay(dataset_id=dataset_id, task="Запись файлов", progress=100)

    stats = {
        "total_chunks": len(chunks),
        "saved": saved,
        "avg_duration": float(np.mean(durations)) if durations else 0,
        "min_duration": float(np.min(durations)) if durations else 0,
        "max_duration": float(np.max(durations)) if durations else 0,
        "allow_short_final": allow_short_final,
    }

    return {
        "status": "success",
        "segments_count": saved,
        "stats": stats,
        "split_points": split_points
    }
