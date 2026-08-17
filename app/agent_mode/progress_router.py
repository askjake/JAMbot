import asyncio, json, time
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/rest/api/v1/agent-mode/progress", tags=["progress"])
_queues = {}

def get_queue(chat_id):
    if chat_id not in _queues:
        _queues[chat_id] = asyncio.Queue(maxsize=200)
    return _queues[chat_id]

def emit(chat_id, event_type, data):
    try:
        get_queue(chat_id).put_nowait({"type": event_type, "data": data, "ts": time.time()})
    except asyncio.QueueFull:
        pass

@router.get("/{chat_id}/stream")
async def stream_progress(chat_id: str):
    async def gen():
        q = get_queue(chat_id)
        nl = chr(10)
        yield "data: " + json.dumps({"type": "connected"}) + nl + nl
        import time as t
        deadline = t.time() + 300
        while t.time() < deadline:
            try:
                ev = await asyncio.wait_for(q.get(), timeout=1.0)
                yield "data: " + json.dumps(ev) + nl + nl
                if ev.get("type") in ("complete", "error"): break
            except asyncio.TimeoutError:
                yield ": heartbeat" + nl + nl
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@router.get("/{chat_id}/status")
async def get_status(chat_id: str):
    return {"chat_id": chat_id, "pending": get_queue(chat_id).qsize()}
