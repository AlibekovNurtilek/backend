#!/bin/bash

# Порты, которые нужно очистить (например, FastAPI)
ports=(8400)

# Остановка процессов по портам
for port in "${ports[@]}"; do
    pid=$(lsof -ti :$port)
    if [ -n "$pid" ]; then
        echo "🔴 Killing process on port $port (PID: $pid)..."
        kill -9 $pid
    else
        echo "🟢 No process running on port $port."
    fi
done

# Остановка Celery воркера по PID-файлу
CELERY_PID_FILE="logs/celery.pid"

if [ -f "$CELERY_PID_FILE" ]; then
    pid=$(cat "$CELERY_PID_FILE")
    if ps -p $pid > /dev/null; then
        echo "🔴 Killing Celery worker (PID: $pid)..."
        kill -9 $pid
    else
        echo "🟢 Celery PID $pid not running."
    fi
    rm -f "$CELERY_PID_FILE"
else
    echo "🟢 No Celery PID file found."
fi
