from enum import Enum

class DatasetStatus(str, Enum):
    INITIALIZING = "INITIALIZING"       # Идёт создание записи и подготовка путей
    SAMPLING = "SAMPLING"               # Идёт сегментация аудио
    SAMPLED = "SAMPLED"                 # Аудио успешно разбито на фрагменты
    TRANSCRIBING = "TRANSCRIBING"       # Идёт автоматическая транскрипция всех сэмплов
    FAILED_TRANSCRIPTION = "FAILED_TRANSCRIPTION"         # Ошибка при транскрипции
    SEMY_TRANSCRIBED = "SEMY_TRANSCRIBED"              # Получен текст из ASR(ЧАСТИЧНО)
    REVIEW = "REVIEW"                   # Ожидает ручную проверку (semi-ready)
    READY = "READY"                     # Все сэмплы проверены и одобрены
    ERROR = "ERROR"                     # Ошибка на любом этапе


class SampleStatus(str, Enum):
    NEW = "NEW"                              # Только что создан, без текста
    TRANSCRIBING = "TRANSCRIBING"            # Отправлен в ASR
    TRANSCRIBED = "TRANSCRIBED"              # Получен текст из ASR
    FAILED_TRANSCRIPTION = "FAILED_TRANSCRIPTION"  # Ошибка при транскрипции
    REJECTED = "REJECTED"                    # Отклонён вручную
    APPROVED = "APPROVED"                    # Одобрен после ручной правки
    # Виртуальный статус (не хранится в БД)
    UNREVIEWED = "UNREVIEWED"