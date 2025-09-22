# app/models/sample_text.py

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Float,
    ForeignKey,
    DateTime,
)
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db import Base
from sqlalchemy import Enum as SqlEnum
from app.models.data_status import SampleStatus

class SampleText(Base):
    __tablename__ = "samples"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)

    filename = Column(String, nullable=True)  # имя .wav файла
    text = Column(Text, nullable=True)        # транскрипция
    duration = Column(Float, nullable=True)   # длина сегмента
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(SqlEnum(SampleStatus), default=SampleStatus.NEW, nullable=False)

    # relationships
    dataset = relationship("AudioDataset", back_populates="samples", passive_deletes=True)
