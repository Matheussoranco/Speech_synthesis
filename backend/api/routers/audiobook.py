"
Audiobook generation
"

from fastapi import APIRouter

router = APIRouter(prefix="/audiobook", tags=["audiobook"])

@router.get("/projects")
async def list_audiobook_projects():
    return []

@router.post("/projects")
async def create_audiobook_project():
    return {"id": "new-project"}

@router.get("/projects/{project_id}")
async def get_audiobook_project(project_id: str):
    return {"id": project_id}

@router.post("/projects/{project_id}/generate")
async def generate_audiobook(project_id: str):
    return {"status": "started"}
