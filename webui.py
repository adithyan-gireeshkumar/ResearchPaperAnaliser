from pathlib import Path

from fastapi.staticfiles import StaticFiles

from main import app


WEBUI_DIR = Path(__file__).parent / "webui"
app.mount("/", StaticFiles(directory=WEBUI_DIR, html=True), name="webui")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("webui:app", host="127.0.0.1", port=8000, reload=False)
