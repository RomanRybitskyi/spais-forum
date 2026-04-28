from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from webapp.runner import PipelineRunner


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Safety Monitor")
runner = PipelineRunner()


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


async def _mjpeg_generator():
    boundary = b"--frame"
    loop = asyncio.get_event_loop()
    while True:
        jpeg = await loop.run_in_executor(
            None, runner.get_latest_jpeg, True, 1.0
        )
        if jpeg is None:
            await asyncio.sleep(0.1)
            continue
        yield (
            boundary + b"\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
            + jpeg + b"\r\n"
        )


@app.get("/video_feed")
def video_feed() -> StreamingResponse:
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


class ScenarioBody(BaseModel):
    scenario: int


class ZoneBody(BaseModel):
    polygon: list[list[int]]


class BoolBody(BaseModel):
    value: bool


@app.get("/api/status")
def get_status() -> dict:
    return runner.status()


@app.post("/api/start")
def start() -> dict:
    runner.start()
    return runner.status()


@app.post("/api/stop")
def stop() -> dict:
    runner.stop()
    return runner.status()


@app.post("/api/scenario")
def set_scenario(body: ScenarioBody) -> dict:
    try:
        runner.set_scenario(body.scenario)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return runner.status()


@app.post("/api/zone")
def set_zone(body: ZoneBody) -> dict:
    try:
        runner.set_zone(body.polygon)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return runner.status()


@app.post("/api/anonymise")
def set_anonymise(body: BoolBody) -> dict:
    runner.set_anonymise(body.value)
    return runner.status()


@app.post("/api/debug")
def set_debug(body: BoolBody) -> dict:
    runner.set_debug(body.value)
    return runner.status()


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
