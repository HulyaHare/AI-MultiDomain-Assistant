import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.lifespan import lifespan
from app.routers.assistant import router as assistant_router

app = FastAPI(
    title="AI Multi-Domain Recommendation Assistant",
    description="Smart assistant powered by local ML models and LLM orchestration — Healthcare, Movies, E-commerce",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(assistant_router)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
