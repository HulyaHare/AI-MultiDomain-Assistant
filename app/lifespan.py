from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.data.loader import load_healthcare, load_movies, load_ecommerce


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("  AI Multi-Domain Assistant — Starting up")
    print("  Loading datasets and fitting ML models (one-time)...")
    print("=" * 60)

    load_healthcare()
    load_movies()
    load_ecommerce()

    print("=" * 60)
    print("  All engines ready! Server is live.")
    print("=" * 60)
    yield
    print("Shutting down...")
