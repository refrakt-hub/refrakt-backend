"""API route handlers"""

import asyncio
import logging
import yaml
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError, TimeoutError as RedisTimeoutError
from models import (
    JobResponse,
    JobStatus,
    JobArtifact,
    JobArtifactsResponse,
    RunResponse,
)
from config import get_settings
from services.ai_service import AIService
from services.r2_service import R2Service
from services.job_service import JobService
from services.websocket_service import WebSocketService
from services.queue import enqueue_training_job
from services.assistant_service import AssistantService
from services.job_context_service import JobContextService
from services.rate_limiter import (
    get_assistant_rate_limit_dependencies,
    get_default_rate_limit_dependencies,
    get_run_rate_limit_dependencies,
)
from services.admission_control import ensure_queue_capacity
from services.job_repository import get_job_repository
from utils import classify_artifact, load_prompt_template
from utils.config_validator import ConfigValidator

logger = logging.getLogger(__name__)

# Initialize services
settings = get_settings()
r2_service = R2Service()
websocket_service = WebSocketService()
job_service = JobService(r2_service, websocket_service)
ai_service = AIService()
job_context_service = JobContextService(job_service)
assistant_service = AssistantService(ai_service, job_context_service)

# Load prompt template
PROMPT_TEMPLATE = load_prompt_template()

# Create router with default rate limiting
# Note: Specific endpoints can override with their own dependencies
_DEFAULT_DEPENDENCIES = get_default_rate_limit_dependencies()
router = APIRouter(dependencies=_DEFAULT_DEPENDENCIES)

_RUN_DEPENDENCIES = [
    *get_run_rate_limit_dependencies(),
    Depends(ensure_queue_capacity),
]

_ASSISTANT_DEPENDENCIES = [
    *get_assistant_rate_limit_dependencies(),
    Depends(ensure_queue_capacity),
]


@router.get("/", dependencies=[])  # Exclude from rate limiting (root endpoint)
async def root():
    """Root endpoint - API information"""
    return {
        "message": "Refrakt Backend API",
        "version": "2.0.0",
        "r2_configured": r2_service.is_configured(),
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
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


@router.get("/test-openai", dependencies=[])  # Exclude from rate limiting (test endpoint)
async def test_openai():
    """Test OpenAI API connection"""
    return ai_service.test_connection()


@router.get("/test-r2", dependencies=[])  # Exclude from rate limiting (test endpoint)
async def test_r2():
    """Test R2 connection and permissions"""
    return r2_service.test_connection()


@router.get("/gpu/usage", dependencies=[])  # Exclude from rate limiting (monitoring endpoint)
async def get_gpu_usage():
    """Get current GPU usage statistics."""
    from config import get_settings
    settings = get_settings()
    
    if not settings.GPU_SCHEDULER_ENABLED:
        return {
            "enabled": False,
            "message": "GPU scheduler is disabled"
        }
    
    from services.gpu_scheduler import get_gpu_scheduler
    scheduler = get_gpu_scheduler()
    return scheduler.get_current_usage()


@router.get("/health", dependencies=[])  # Exclude from rate limiting (health check)
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    Returns 200 if healthy, 503 if unhealthy.
    Checks Redis connectivity and queue accessibility.
    This endpoint is excluded from rate limiting.
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "checks": {}
    }
    overall_healthy = True
    
    # Check Redis connectivity
    redis_check = {"status": "ok"}
    if not settings.QUEUE_URL:
        redis_check["status"] = "error"
        redis_check["error"] = "QUEUE_URL not configured"
        overall_healthy = False
    else:
        try:
            # Create a test Redis connection with timeout
            test_redis = Redis.from_url(
                settings.QUEUE_URL,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            # Try to ping Redis
            await asyncio.wait_for(test_redis.ping(), timeout=2.0)
            await test_redis.aclose()
            redis_check["status"] = "ok"
        except (RedisConnectionError, RedisTimeoutError, asyncio.TimeoutError) as e:
            redis_check["status"] = "error"
            redis_check["error"] = str(e)
            overall_healthy = False
        except Exception as e:
            redis_check["status"] = "error"
            redis_check["error"] = f"Unexpected error: {str(e)}"
            overall_healthy = False
    
    health_status["checks"]["redis"] = redis_check
    
    # Check queue/job repository accessibility
    queue_check = {"status": "ok"}
    if not settings.QUEUE_URL:
        queue_check["status"] = "error"
        queue_check["error"] = "QUEUE_URL not configured"
        overall_healthy = False
    else:
        try:
            repository = get_job_repository()
            # Try to list jobs (with limit to keep it fast)
            await asyncio.wait_for(
                asyncio.to_thread(repository.list_jobs, limit=1),
                timeout=2.0
            )
            queue_check["status"] = "ok"
        except asyncio.TimeoutError:
            queue_check["status"] = "error"
            queue_check["error"] = "Queue access timeout"
            overall_healthy = False
        except Exception as e:
            queue_check["status"] = "error"
            queue_check["error"] = str(e)
            overall_healthy = False
    
    health_status["checks"]["queue"] = queue_check
    
    # Update overall status
    if not overall_healthy:
        health_status["status"] = "unhealthy"
    
    # Return appropriate status code
    status_code = status.HTTP_200_OK if overall_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(content=health_status, status_code=status_code)


async def _start_training_job(
    prompt_text: str,
    user_id: str,
    dataset_upload: Optional[UploadFile] = None,
    *,
    source: str = "api",
) -> JobResponse:
    job_service.purge_expired_datasets()

    job_id = job_service.create_job(prompt_text, user_id)
    try:
        job_service.update_job_status(job_id, "generating")

        dataset_metadata = None
        if dataset_upload is not None:
            try:
                dataset_metadata = job_service.stage_dataset(job_id, dataset_upload)
            except ValueError as staging_error:
                job_service.update_job_status(job_id, "error", error=str(staging_error))
                raise HTTPException(status_code=400, detail=str(staging_error))
            except Exception as staging_error:
                job_service.update_job_status(job_id, "error", error=str(staging_error))
                raise HTTPException(status_code=500, detail=str(staging_error))

        dataset_hint = "DATASET_UPLOAD: present" if dataset_metadata else "DATASET_UPLOAD: none"
        if dataset_metadata and dataset_metadata.get("num_classes"):
            dataset_hint += f" | NUM_CLASSES: {dataset_metadata['num_classes']}"

        try:
            config = ai_service.generate_yaml_config(
                prompt_text,
                PROMPT_TEMPLATE,
                dataset_hint=dataset_hint,
            )
        except ValueError as e:
            job_service.update_job_status(job_id, "error", error=str(e))
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            job_service.update_job_status(job_id, "error", error=str(e))
            raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

        if dataset_metadata:
            dataset_cfg = config.get("dataset") if isinstance(config, dict) else None
            if not isinstance(dataset_cfg, dict):
                dataset_cfg = {}
            dataset_cfg.setdefault("name", "custom")
            params_cfg = dataset_cfg.get("params")
            if not isinstance(params_cfg, dict):
                params_cfg = {}
            params_cfg["zip_path"] = dataset_metadata["path"]
            params_cfg.setdefault("task_type", "supervised")
            dataset_cfg["params"] = params_cfg
            config["dataset"] = dataset_cfg

            num_classes = dataset_metadata.get("num_classes")
            if num_classes:
                model_cfg = config.get("model") if isinstance(config, dict) else None
                if not isinstance(model_cfg, dict):
                    model_cfg = {}
                params_model = model_cfg.get("params")
                if not isinstance(params_model, dict):
                    params_model = {}
                params_model["num_classes"] = int(num_classes)
                model_cfg["params"] = params_model
                config["model"] = model_cfg

        job_service.update_job_status(job_id, "generating", config=config)

        job_dir = settings.JOBS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        # Persist the generated configuration as config.yaml inside the job directory.
        config_path = job_dir / "config.yaml"
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

        # Store metadata (including the on-disk path) for observability; the worker
        # will reconstruct the path from JOBS_DIR + job_id to avoid host/container
        # path mismatches.
        job_service.record_job_configuration(job_id, config, str(config_path))

        queue_job_id = await enqueue_training_job(job_id, source=source)
        job_service.update_job_status(job_id, "queued", queue_job_id=queue_job_id)

        return JobResponse(
            job_id=job_id,
            status="queued",
            message="Job queued successfully",
            queue_job_id=queue_job_id,
        )
    except HTTPException:
        raise
    except Exception as exc:
        job_service.update_job_status(job_id, "error", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Error running job: {str(exc)}")


@router.post(
    "/run",
    response_model=RunResponse,
    dependencies=_RUN_DEPENDENCIES,
)
async def run_job(
    request: Request,
    prompt: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    dataset: Optional[UploadFile] = File(None),
    conversation_id: Optional[str] = Form(None),
):
    """Run complete pipeline: prompt → YAML → training → R2 upload"""
    content_type = request.headers.get("content-type", "")
    request_prompt = prompt
    request_user_id = user_id or "anonymous"
    dataset_upload = dataset

    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception as parse_error:
            raise HTTPException(status_code=400, detail=f"Invalid JSON body: {parse_error}")
        request_prompt = body.get("prompt")
        request_user_id = body.get("user_id", "anonymous")
        conversation_id = body.get("conversation_id") or conversation_id
        dataset_upload = None

    if not request_prompt:
        raise HTTPException(status_code=400, detail="'prompt' field is required")

    assistant_result = None
    effective_prompt = request_prompt

    # By default, assume we're *not* in training mode until proven otherwise
    resolved_training_intent = False

    if settings.ASSISTANT_ENABLED:
        assistant_result = assistant_service.process_message(
            message=request_prompt,
            conversation_id=conversation_id,
            user_id=request_user_id,
        )
        conversation_id = assistant_result.conversation_id

        # Primary signal from assistant
        if assistant_result.intent == "training_request":
            resolved_training_intent = True
            if assistant_result.training_prompt:
                effective_prompt = assistant_result.training_prompt
        else:
            # Backend heuristic: treat clearly-training prompts that mention a supported model
            # as training requests, even if the assistant intent is not "training_request".
            validator = ConfigValidator()
            supported_ref = validator.find_supported_model_reference(request_prompt)
            lowered = request_prompt.lower()
            has_training_verb = any(
                phrase in lowered
                for phrase in (
                    "train ",
                    "train\n",
                    "fine-tune",
                    "fine tune",
                    "run training",
                    "start training",
                    "fit a model",
                    "fit the model",
                )
            )

            if supported_ref and has_training_verb:
                logger.info(
                    "Overriding assistant intent to 'training_request' based on backend "
                    "heuristic (alias='%s', canonical='%s')",
                    supported_ref[0],
                    supported_ref[1],
                )
                resolved_training_intent = True

    if not resolved_training_intent:
        # Stay in assistant mode – no training job will be started
        return RunResponse(
            mode="assistant",
            message=assistant_result.reply if assistant_result else "I am still gathering details on that.",
            job=None,
            conversation_id=conversation_id,
            intent=assistant_result.intent if assistant_result else None,
            confidence=assistant_result.confidence if assistant_result else None,
            training_prompt=assistant_result.training_prompt if assistant_result else None,
        )

    job_payload = await _start_training_job(effective_prompt, request_user_id, dataset_upload)

    return RunResponse(
        mode="job",
        message=assistant_result.reply if assistant_result else "Job queued successfully",
        job=job_payload,
        conversation_id=conversation_id,
        intent=assistant_result.intent if assistant_result else "training_request",
        confidence=assistant_result.confidence if assistant_result else None,
        training_prompt=assistant_result.training_prompt if assistant_result else effective_prompt,
    )


@router.post(
    "/assistant",
    dependencies=_ASSISTANT_DEPENDENCIES,
    deprecated=True,
)
async def assistant_endpoint(payload: dict):
    """
    DEPRECATED: Use /run endpoint instead.
    
    This endpoint is deprecated and will be removed in a future version.
    The /run endpoint provides the same functionality with additional features:
    - Supports file uploads (dataset attachments)
    - Unified response format
    - Better error handling
    
    Migration: Replace POST /assistant with POST /run using form data or JSON.
    """
    message = payload.get("message")
    user_id = payload.get("user_id") or "anonymous"
    conversation_id = payload.get("conversation_id")

    if not message:
        raise HTTPException(status_code=400, detail="'message' field is required")

    if not settings.ASSISTANT_ENABLED:
        raise HTTPException(status_code=503, detail="Assistant features are disabled")

    result = assistant_service.process_message(
        message=message,
        conversation_id=conversation_id,
        user_id=user_id,
    )

    response = {
        "intent": result.intent,
        "message": result.reply,
        "training_prompt": result.training_prompt,
        "confidence": result.confidence,
        "conversation_id": result.conversation_id,
    }

    if result.intent == "training_request":
        prompt_text = result.training_prompt or message
        job_payload = await _start_training_job(prompt_text, user_id, source="assistant")
        response["job"] = job_payload.model_dump()

    return response


@router.post("/assistant/reindex")
async def assistant_reindex():
    """
    Rebuild the assistant's RAG (Retrieval-Augmented Generation) knowledge index.
    
    This endpoint rebuilds the vector store used for document retrieval in assistant conversations.
    It indexes:
    - Prompt template documentation
    - Configuration examples (from configs/ directory)
    - Supported models summary
    
    Returns detailed statistics about the reindexing operation including:
    - Number of chunks indexed
    - Sources indexed
    - Time taken
    - Embedding statistics
    """
    if not settings.ASSISTANT_ENABLED or not settings.ASSISTANT_RETRIEVAL_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="Assistant retrieval disabled. Enable ASSISTANT_ENABLED and ASSISTANT_RETRIEVAL_ENABLED to use this endpoint."
        )

    success, details = assistant_service.rebuild_static_index()
    
    if not success:
        error_type = details.get("error_type", "unknown") if details else "unknown"
        error_message = details.get("error_message", "Unknown error") if details else "Unknown error"
        status_code = 500
        
        # Provide more specific error codes
        if error_type == "configuration":
            status_code = 503  # Service unavailable
        elif error_type == "empty_corpus":
            status_code = 400  # Bad request (no documents to index)
        
        raise HTTPException(
            status_code=status_code,
            detail={
                "error": details.get("error", "Failed to rebuild assistant index") if details else "Failed to rebuild assistant index",
                "error_type": error_type,
                "error_message": error_message,
                "details": details,
            }
        )

    return details


@router.websocket("/ws/job/{job_id}/logs")  # WebSocket endpoints don't use rate limiting dependencies
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
        # Keep connection alive
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.send_text("__ping__")
    
    except WebSocketDisconnect:
        await websocket_service.remove_connection(job_id, websocket)
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {str(e)}", exc_info=True)
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
        logger.debug(f"Error creating JobStatus for job {job_id}: {str(e)}", exc_info=True)
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
                    logger.warning(f"Error processing file {file_path}: {str(e)}", exc_info=True)
    
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

