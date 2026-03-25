"""Aggregate API routers."""

from fastapi import APIRouter

from app.api.routes import compare, health, reports

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(compare.router)
api_router.include_router(reports.router)
