import os
from fastapi import FastAPI
from app.routes import audio_route, auth_route, dataset_route, sample_route, transcription_router, ws_router, user_route
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app.db import Base, engine
from app.auth import models  # обязательно, чтобы модель подгрузилась
from app.routes.ws_router import redis_listener

# создаёт таблицы, если их нет
Base.metadata.create_all(bind=engine)

app = FastAPI(title="TTS Audio Validator2 ")
app.include_router(ws_router.router)

@app.on_event("startup")
async def startup():
    import asyncio
    asyncio.create_task(redis_listener())


# 🔓 Настройка CORS — разрешить всё
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://80.72.180.130:8440",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_route.router)
app.include_router(audio_route.router)
app.include_router(dataset_route.router)
app.include_router(sample_route.router)
app.include_router(transcription_router.router)
app.include_router(user_route.router)

@app.get("/")
def read_root():
    return {"Apllication running"}
