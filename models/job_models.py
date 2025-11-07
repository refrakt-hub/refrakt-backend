"""Job-related Pydantic models"""

from typing import Optional, List
from pydantic import BaseModel


class JobRequest(BaseModel):
    """Request model for creating a new job"""
    prompt: str
    user_id: Optional[str] = "anonymous"


class JobResponse(BaseModel):
    """Response model for job creation"""
    job_id: str
    status: str
    message: str


class JobStatus(BaseModel):
    """Job status model"""
    job_id: str
    status: str
    created_at: str
    updated_at: str
    config: Optional[dict] = None
    result_path: Optional[str] = None
    r2_uploaded: Optional[bool] = False
    error: Optional[str] = None
    logs: Optional[List[str]] = None
    dataset: Optional[dict] = None


class JobArtifact(BaseModel):
    """Job artifact model"""
    name: str
    path: str
    type: str  # 'model', 'log', 'visualization', 'checkpoint', 'other'
    size: int
    modified: str
    download_url: Optional[str] = None  # R2 presigned URL
    public_url: Optional[str] = None     # R2 public URL


class JobArtifactsResponse(BaseModel):
    """Response model for job artifacts"""
    job_id: str
    artifacts: List[JobArtifact]
    r2_uploaded: bool

