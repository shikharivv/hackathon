from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from src.config.settings import get_settings


# ------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ------------------------------------------------------------------
@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise shared resources on startup and tear them down on shutdown."""

    settings = get_settings()
    logger.info("Starting {} v{}", settings.app_name, settings.version)

    from src.rag.runtime import PolicyRuntime
    runtime = PolicyRuntime()
    app.state.rag_chain = runtime
    app.state.vectorstore = runtime
    app.state.documents_indexed = runtime.documents_indexed
    app.state.llm = runtime
    logger.info("Startup complete")
    yield

    # -- Shutdown ----------------------------------------------------
    logger.info("Shutting down {}", settings.app_name)


# ------------------------------------------------------------------
# App factory
# ------------------------------------------------------------------
def create_app() -> FastAPI:
    """Build and return the configured :class:`FastAPI` application."""

    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        lifespan=_lifespan,
    )

    # -- CORS --------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- Routes ------------------------------------------------------
    from src.api.routes import router

    app.include_router(router)

    # -- Error handlers ----------------------------------------------
    @app.exception_handler(ValueError)
    async def _value_error_handler(
        _request: Request, exc: ValueError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"detail": str(exc)},
        )

    @app.exception_handler(FileNotFoundError)
    async def _file_not_found_handler(
        _request: Request, exc: FileNotFoundError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={"detail": str(exc)},
        )

    @app.exception_handler(Exception)
    async def _generic_error_handler(
        _request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception("Unhandled exception")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    return app


app = create_app()
