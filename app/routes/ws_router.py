# ws_router.py
from fastapi import WebSocket, APIRouter
import redis.asyncio as aioredis
import json
import asyncio

router = APIRouter()
connections = {}

@router.websocket("/ws/{dataset_id}")
async def websocket_endpoint(websocket: WebSocket, dataset_id: int):
    await websocket.accept()
    dataset_id = int(dataset_id)
    connections.setdefault(dataset_id, []).append(websocket)

    try:
        while True:
            await websocket.receive_text()
    except:
        connections[dataset_id].remove(websocket)

async def redis_listener():
    try:
        print("🟢 Redis listener запускается...")
        redis = aioredis.Redis(host="localhost", port=6379, decode_responses=True)
        pubsub = redis.pubsub()
        await pubsub.subscribe("ws_progress")
        print("✅ Redis подписан на канал ws_progress")

        while True:
            try:
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message:
                    data = json.loads(message["data"])
                    dataset_id = data["dataset_id"]
                    ws_list = connections.get(dataset_id, [])
                    for ws in ws_list:
                        try:
                            await ws.send_json(data)
                        except Exception as e:
                            print(f"❌ Ошибка при отправке WS: {e}")
                await asyncio.sleep(0.01)
            except Exception as inner:
                print(f"⚠️ Ошибка в loop: {inner}")
                await asyncio.sleep(1)
    except Exception as outer:
        print(f"❌ Ошибка запуска redis_listener: {outer}")
