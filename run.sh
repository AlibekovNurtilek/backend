#!/bin/bash

# Название лог-файлов и PID-файлов
LOG_DIR="logs"
SERVER_LOG="$LOG_DIR/server.log"
CELERY_LOG="$LOG_DIR/celery.log"
CELERY_PID_FILE="$LOG_DIR/celery.pid"

# Создаём папку для логов, если не существует
mkdir -p "$LOG_DIR"

# Запускаем FastAPI сервер (stdout + stderr в один лог)
nohup uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8400 \
  --reload \
  > "$SERVER_LOG" 2>&1 &

echo "✅ Server started on port 8400 (logs: $SERVER_LOG)"

# Запускаем Celery воркер и сохраняем его PID
nohup celery -A app.celery_worker worker \
  --loglevel=info \
  --concurrency=16 \
  -Q yt_asr_data > "$CELERY_LOG" 2>&1 &

echo $! > "$CELERY_PID_FILE"
echo "✅ Celery worker started (logs: $CELERY_LOG, pid: $(cat $CELERY_PID_FILE))"
