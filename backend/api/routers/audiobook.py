"""
Audiobook generation
"""

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/audiobook", tags=["audiobook"])

@router.get("/projects")
async def list_audiobook_projects():
    return []

@router.post("/projects")
async def create_audiobook_project():
    raise HTTPException(503, "Audiobook project storage is not implemented")

@router.get("/projects/{project_id}")
async def get_audiobook_project(project_id: str):
    raise HTTPException(404, "Audiobook project not found")

@router.post("/projects/{project_id}/generate")
async def generate_audiobook(project_id: str):
    raise HTTPException(503, "No real audiobook pipeline is configured")
