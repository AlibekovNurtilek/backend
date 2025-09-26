from pydantic import BaseModel, Field
from datetime import datetime

class DatasetBase(BaseModel):
    name: str
    url: str
    source_rel_path: str
    segments_rel_dir: str
    count_of_samples: int
    duration: float | None = None
    status: str

class DatasetCreate(DatasetBase):
    pass

class DatasetUpdate(DatasetBase):
    pass

class DatasetOut(DatasetBase):
    id: int
    created_at: datetime
    last_update: datetime

    class Config:
        orm_mode = True

class DatasetInitRequest(BaseModel):
    url: str
    min_duration: int | None = None
    max_duration: int | None = None

from typing import List

class DatasetListResponse(BaseModel):
    items: List[DatasetOut]
    total: int
    page: int | None = None
    limit: int | None = None

class DatasetImageUpdate(BaseModel):
    dataset_img: str


class DatasetResegmentRequest(BaseModel):
    min_duration: float = Field(..., gt=0, description="Минимальная длительность сегмента в секундах")
    max_duration: float = Field(..., gt=0, description="Максимальная длительность сегмента в секундах")
    
    class Config:
        json_schema_extra = {
            "example": {
                "min_duration": 1.0,
                "max_duration": 10.0
            }
        }