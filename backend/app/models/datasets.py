from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db import Base
from sqlalchemy import Enum as SqlEnum
from app.models.data_status import DatasetStatus

class AudioDataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    url = Column(String, nullable=False)

    # Новый путь к полному .wav файлу
    source_rel_path = Column(String, nullable=True)  # 🔹 Например: datasets/yt_.../full.wav
    segments_rel_dir = Column(String, nullable=False)     # 🔹 Например: datasets/yt_.../samples/
    
    count_of_samples = Column(Integer, default=0)
    duration = Column(Float, nullable=True)  # в секундах
    created_at = Column(DateTime, default=datetime.utcnow)
    last_update = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)    
    status = Column(SqlEnum(DatasetStatus), default=DatasetStatus.INITIALIZING, nullable=False)

    # relationships
    samples = relationship(
        "SampleText",
        back_populates="dataset",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

