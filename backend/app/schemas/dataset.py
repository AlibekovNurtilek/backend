from pydantic import BaseModel
from datetime import datetime

class DatasetBase(BaseModel):
    name: str
    url: str
    source_rel_path: str
    segments_rel_dir: str
    count_of_samples: int
    duration: float | None = None
    status: str
    dataset_img :str | None = None

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
    min_length: int | None = None
    max_length: int | None = None

from typing import List

class DatasetListResponse(BaseModel):
    items: List[DatasetOut]
    total: int

class DatasetImageUpdate(BaseModel):
    dataset_img: str
