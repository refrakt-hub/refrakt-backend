"""Job-related Pydantic models"""

from typing import Optional, List, Literal
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
    queue_job_id: Optional[str] = None


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
    queue_job_id: Optional[str] = None


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


class RunResponse(BaseModel):
    """Unified response for /run endpoint covering assistant and job paths."""

    mode: Literal["assistant", "job"]
    message: str
    job: Optional[JobResponse] = None
    conversation_id: Optional[str] = None
    # Assistant metadata (always present when assistant is enabled)
    intent: Optional[str] = None  # "training_request", "general", "unknown"
    confidence: Optional[float] = None  # 0.0-1.0 confidence score
    training_prompt: Optional[str] = None  # Refined prompt if training intent detected

