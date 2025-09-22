from celery import Celery

celery_app = Celery(
    "asr_tasks",
    broker="redis://localhost:6379/5",   # <— своя БД (было :0)
    backend="redis://localhost:6379/5",
)

# пусть все твои задачи по умолчанию публикуются в свою очередь
celery_app.conf.task_default_queue = "yt_asr_data"
celery_app.conf.task_create_missing_queues = True
# (опционально) префиксы ключей, чтобы в Redis было чище и не пересекалось
celery_app.conf.broker_transport_options = {"global_keyprefix": "yt_asr_data:"}
celery_app.conf.result_backend_transport_options = {"global_keyprefix": "yt_asr_data:"}
