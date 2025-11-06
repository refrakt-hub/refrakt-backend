"""API route handlers"""

import asyncio
import tempfile
import yaml
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from models import (
    JobRequest,
    JobResponse,
    JobStatus,
    JobArtifact,
    JobArtifactsResponse,
)
from config import get_settings
from services.ai_service import AIService
from services.r2_service import R2Service
from services.job_service import JobService
from services.websocket_service import WebSocketService
from utils import classify_artifact, load_prompt_template

# Initialize services
settings = get_settings()
r2_service = R2Service()
websocket_service = WebSocketService()
job_service = JobService(r2_service, websocket_service)
ai_service = AIService()

# Load prompt template
PROMPT_TEMPLATE = load_prompt_template()

# Create router
router = APIRouter()


@router.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "message": "Refrakt Backend API",
        "version": "2.0.0",
        "r2_configured": r2_service.is_configured(),
        "docs": "/docs",
        "endpoints": {
            "run_job": "/run",
            "job_status": "/job/{job_id}",
            "jobs_list": "/jobs",
            "job_artifacts": "/job/{job_id}/artifacts",
            "download_artifact_presigned": "/job/{job_id}/download/{artifact_path}",
            "logs_websocket": "/ws/job/{job_id}/logs",
            "test_openai": "/test-openai",
            "test_r2": "/test-r2"
        }
    }


@router.get("/test-openai")
async def test_openai():
    """Test OpenAI API connection"""
    return ai_service.test_connection()


@router.get("/test-r2")
async def test_r2():
    """Test R2 connection and permissions"""
    return r2_service.test_connection()


@router.post("/run", response_model=JobResponse)
async def run_job(request: JobRequest):
    """Run complete pipeline: prompt → YAML → training → R2 upload"""
    job_id = job_service.create_job(request.prompt, request.user_id)
    
    try:
        # Update job status
        job_service.update_job_status(job_id, "generating")
        
        # Generate YAML using OpenAI
        try:
            config = ai_service.generate_yaml_config(request.prompt, PROMPT_TEMPLATE)
            print(f"DEBUG: Config generated successfully!")
            print(f"DEBUG: Config keys: {list(config.keys()) if config else 'None'}")
        except ValueError as e:
            job_service.update_job_status(job_id, "error", error=str(e))
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            print(f"DEBUG: OpenAI API error: {str(e)}")
            job_service.update_job_status(job_id, "error", error=str(e))
            raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
        
        # Save config to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name
        
        # Start training in background
        asyncio.create_task(job_service.run_job(job_id, config, config_path))
        
        return JobResponse(
            job_id=job_id,
            status="running",
            message="Job started successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        job_service.update_job_status(job_id, "error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error running job: {str(e)}")


@router.websocket("/ws/job/{job_id}/logs")
async def websocket_logs(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time log streaming"""
    await websocket.accept()
    
    job = job_service.get_job(job_id)
    if not job:
        await websocket.send_text(f"Error: Job {job_id} not found")
        await websocket.close()
        return
    
    # Add connection to active connections
    await websocket_service.add_connection(job_id, websocket)
    
    try:
        # Send existing logs
        if "logs" in job:
            for log_line in job["logs"]:
                await websocket.send_text(log_line)
        
        # Keep connection alive
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.send_text("__ping__")
    
    except WebSocketDisconnect:
        await websocket_service.remove_connection(job_id, websocket)
    except Exception as e:
        print(f"WebSocket error for job {job_id}: {str(e)}")
        await websocket_service.remove_connection(job_id, websocket)


@router.get("/job/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get job status by ID"""
    job = job_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = job.copy()
    
    # Limit logs to last 100 lines for status endpoint
    if "logs" in job_data and len(job_data["logs"]) > 100:
        job_data["logs"] = job_data["logs"][-100:]
    
    try:
        return JobStatus(**job_data)
    except Exception as e:
        print(f"DEBUG: Error creating JobStatus for job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating job status: {str(e)}")


@router.get("/jobs")
async def list_jobs():
    """List all jobs"""
    return {"jobs": job_service.list_jobs()}


@router.get("/job/{job_id}/artifacts", response_model=JobArtifactsResponse)
async def list_job_artifacts(job_id: str):
    """List all artifacts with R2 URLs"""
    job = job_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_dir = settings.JOBS_DIR / job_id
    artifacts = []
    r2_uploaded = job.get("r2_uploaded", False)
    local_cleaned = job.get("local_cleaned", False)
    
    # If local files were cleaned up, use stored metadata
    if local_cleaned and "artifact_metadata" in job:
        artifact_metadata = job["artifact_metadata"]
        for meta in artifact_metadata:
            # Generate URLs if R2 is configured
            download_url = None
            public_url = None
            
            if r2_uploaded and r2_service.is_configured():
                download_url = r2_service.generate_presigned_url(
                    job_id,
                    meta["path"],
                    expiration=3600
                )
                public_url = f"{settings.R2_PUBLIC_URL}/jobs/{job_id}/{meta['path']}"
            
            # Classify artifact type from path
            artifact_path = Path(meta["path"])
            artifact_type = classify_artifact(artifact_path)
            
            artifact = JobArtifact(
                name=meta["name"],
                path=meta["path"],
                type=artifact_type,
                size=meta["size"],
                modified=meta["modified"],
                download_url=download_url,
                public_url=public_url
            )
            artifacts.append(artifact)
    elif job_dir.exists():
        # Local files still exist, read from disk
        for file_path in job_dir.rglob('*'):
            if file_path.is_file():
                try:
                    stat = file_path.stat()
                    relative_path = str(file_path.relative_to(job_dir))
                    
                    # Generate URLs if R2 is configured
                    download_url = None
                    public_url = None
                    
                    if r2_uploaded and r2_service.is_configured():
                        download_url = r2_service.generate_presigned_url(
                            job_id,
                            relative_path,
                            expiration=3600
                        )
                        public_url = f"{settings.R2_PUBLIC_URL}/jobs/{job_id}/{relative_path}"
                    
                    artifact = JobArtifact(
                        name=file_path.name,
                        path=relative_path,
                        type=classify_artifact(file_path),
                        size=stat.st_size,
                        modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        download_url=download_url,
                        public_url=public_url
                    )
                    artifacts.append(artifact)
                except Exception as e:
                    print(f"Error processing file {file_path}: {str(e)}")
    
    return JobArtifactsResponse(
        job_id=job_id,
        artifacts=artifacts,
        r2_uploaded=r2_uploaded
    )


@router.get("/job/{job_id}/download/{artifact_path:path}")
async def get_download_url(job_id: str, artifact_path: str):
    """Get presigned download URL for an artifact"""
    job = job_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if not job.get("r2_uploaded", False):
        raise HTTPException(status_code=400, detail="Artifacts not uploaded to R2 yet")
    
    if not r2_service.is_configured():
        raise HTTPException(status_code=503, detail="R2 not configured")
    
    # Generate presigned URL (valid for 1 hour)
    download_url = r2_service.generate_presigned_url(job_id, artifact_path, expiration=3600)
    
    if not download_url:
        raise HTTPException(status_code=500, detail="Failed to generate download URL")
    
    return {
        "download_url": download_url,
        "expires_in": 3600,
        "public_url": f"{settings.R2_PUBLIC_URL}/jobs/{job_id}/{artifact_path}"
    }

