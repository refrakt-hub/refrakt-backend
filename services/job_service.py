"""Job management service"""

import asyncio
import os
import shutil
import uuid
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Set

from fastapi import UploadFile

from config import get_settings
from services.r2_service import R2Service
from services.websocket_service import WebSocketService

DATASET_TTL_SECONDS = 3600
DATASET_ROOT = Path("/tmp/datasets")


class JobService:
    """Service for managing ML training jobs"""
    
    def __init__(self, r2_service: R2Service, websocket_service: WebSocketService):
        """Initialize job service"""
        self.settings = get_settings()
        self.r2_service = r2_service
        self.websocket_service = websocket_service
        self.jobs: Dict[str, dict] = {}
        self.dataset_root = DATASET_ROOT
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        self.dataset_ttl_seconds = DATASET_TTL_SECONDS
    
    def create_job(self, prompt: str, user_id: str = "anonymous") -> str:
        """
        Create a new job
        
        Args:
            prompt: User prompt
            user_id: User identifier
            
        Returns:
            Job ID
        """
        job_id = str(uuid.uuid4())
        
        self.jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "prompt": prompt,
            "user_id": user_id,
            "logs": [],
            "r2_uploaded": False,
            "dataset": None,
        }
        
        return job_id

    def purge_expired_datasets(self):
        """Remove dataset directories that have exceeded their TTL."""
        now = datetime.utcnow()
        for job_id, job_data in list(self.jobs.items()):
            dataset_meta = job_data.get("dataset")
            if not dataset_meta:
                continue
            expires_at = dataset_meta.get("expires_at")
            if not expires_at:
                continue
            try:
                expires_dt = datetime.fromisoformat(expires_at)
            except ValueError:
                continue
            if expires_dt <= now and job_data.get("status") != "running":
                dir_path = Path(dataset_meta.get("dir", ""))
                self._delete_dataset_dir(dir_path)
                job_data["dataset"] = None

    def stage_dataset(self, job_id: str, upload: UploadFile) -> dict:
        """Persist an uploaded dataset zip for a job."""
        if job_id not in self.jobs:
            raise ValueError(f"Unknown job_id: {job_id}")
        filename = upload.filename or "dataset.zip"
        if not filename.lower().endswith(".zip"):
            raise ValueError("Uploaded dataset must be a .zip file")

        job_dataset_dir = self.dataset_root / job_id
        job_dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = job_dataset_dir / "source.zip"

        try:
            upload.file.seek(0)
            with dataset_path.open("wb") as destination:
                shutil.copyfileobj(upload.file, destination)
        except Exception as write_error:
            self._delete_dataset_dir(job_dataset_dir)
            raise ValueError(f"Failed to save uploaded dataset: {write_error}") from write_error
        finally:
            try:
                upload.file.close()
            except Exception:
                pass

        now = datetime.utcnow()
        expires = now + timedelta(seconds=self.dataset_ttl_seconds)
        metadata = {
            "original_name": filename,
            "dir": str(job_dataset_dir),
            "path": str(dataset_path),
            "created_at": now.isoformat(),
            "expires_at": expires.isoformat(),
        }

        analysis = self._analyze_dataset_zip(dataset_path)
        if analysis.get("num_classes"):
            metadata["num_classes"] = analysis["num_classes"]
            metadata["class_names"] = analysis.get("class_names")

        self.jobs[job_id]["dataset"] = metadata
        return metadata

    def ensure_dataset_active(self, job_id: str):
        """Ensure a staged dataset is still valid before training."""
        dataset_meta = self.jobs.get(job_id, {}).get("dataset")
        if not dataset_meta:
            return

        expires_at = dataset_meta.get("expires_at")
        if expires_at:
            try:
                expires_dt = datetime.fromisoformat(expires_at)
            except ValueError:
                expires_dt = None
            if expires_dt and expires_dt <= datetime.utcnow():
                self._delete_dataset_dir(Path(dataset_meta.get("dir", "")))
                self.jobs[job_id]["dataset"] = None
                raise ValueError("Dataset expired before training could start")

        dataset_path = Path(dataset_meta.get("path", ""))
        if not dataset_path.exists():
            raise FileNotFoundError("Staged dataset file no longer exists")

    def _delete_dataset_dir(self, dir_path: Path):
        if dir_path.exists():
            shutil.rmtree(dir_path, ignore_errors=True)

    def _analyze_dataset_zip(self, dataset_path: Path) -> Dict[str, Optional[object]]:
        """Inspect the dataset zip to infer class names/count for supervised tasks."""
        class_names: Set[str] = set()
        keywords_train = {"train", "training", "traing", "trainging"}
        keywords_supervised = keywords_train | {"val", "test", "validation", "testing"}
        image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

        try:
            with zipfile.ZipFile(dataset_path, "r") as zip_file:
                for info in zip_file.infolist():
                    if info.is_dir():
                        continue
                    suffix = Path(info.filename).suffix.lower()
                    if suffix not in image_suffixes:
                        continue
                    parts = [part for part in info.filename.split("/") if part]
                    if len(parts) < 3:
                        continue
                    for idx, segment in enumerate(parts[:-1]):
                        segment_lower = segment.lower()
                        if segment_lower in keywords_train and idx + 1 < len(parts) - 0:
                            class_names.add(parts[idx + 1])
                            break
                        if segment_lower in keywords_supervised and idx + 1 < len(parts) - 0:
                            class_names.add(parts[idx + 1])
                            break
        except Exception as analysis_error:
            print(f"DEBUG: Failed to analyze dataset zip {dataset_path}: {analysis_error}")

        ordered_class_names = sorted(class_names)
        return {
            "num_classes": len(ordered_class_names) if ordered_class_names else None,
            "class_names": ordered_class_names if ordered_class_names else None,
        }
    
    def update_job_status(
        self,
        job_id: str,
        status: str,
        config: Optional[dict] = None,
        error: Optional[str] = None
    ):
        """Update job status"""
        if job_id not in self.jobs:
            return
        
        self.jobs[job_id]["status"] = status
        self.jobs[job_id]["updated_at"] = datetime.now().isoformat()
        
        if config is not None:
            self.jobs[job_id]["config"] = config
        if error is not None:
            self.jobs[job_id]["error"] = error
    
    def add_log(self, job_id: str, log_line: str):
        """Add log line to job"""
        if job_id not in self.jobs:
            return
        
        if "logs" not in self.jobs[job_id]:
            self.jobs[job_id]["logs"] = []
        
        self.jobs[job_id]["logs"].append(log_line)
    
    def get_job(self, job_id: str) -> Optional[dict]:
        """Get job by ID"""
        return self.jobs.get(job_id)
    
    def list_jobs(self) -> list:
        """List all jobs"""
        return list(self.jobs.values())
    
    async def run_job(
        self,
        job_id: str,
        config: dict,
        config_path: str
    ):
        """
        Run refrakt CLI job in background
        
        Args:
            job_id: Job identifier
            config: Parsed YAML config
            config_path: Path to YAML config file
        """
        try:
            # Create output directory
            output_dir = self.settings.JOBS_DIR / job_id
            output_dir.mkdir(exist_ok=True)
            
            # Update job status
            self.update_job_status(job_id, "running", config=config)
            
            # Run refrakt CLI
            # Use relative path from project root for log-dir
            # Project root is where refrakt CLI runs from
            project_root = self.settings.PROJECT_ROOT
            log_dir_relative = os.path.relpath(output_dir, project_root)
            
            # Verify config file exists
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")
            
            # Verify project root exists
            if not project_root.exists():
                raise FileNotFoundError(f"Project root not found: {project_root}")
            
            try:
                self.ensure_dataset_active(job_id)
            except Exception as dataset_error:
                self.update_job_status(job_id, "error", error=str(dataset_error))
                await self.websocket_service.broadcast_log(job_id, f"[Dataset] {dataset_error}")
                print(f"DEBUG: Job {job_id} dataset error: {dataset_error}")
                return

            # Find refrakt command - try to locate it in PATH or use python -m
            refrakt_cmd = shutil.which("refrakt")
            if not refrakt_cmd:
                # Fallback: try using python -m refrakt_cli
                python_cmd = shutil.which("python3") or shutil.which("python")
                if python_cmd:
                    cmd = [
                        python_cmd,
                        "-m", "refrakt_cli",
                        "--config", config_path,
                        "--log-dir", log_dir_relative
                    ]
                else:
                    raise FileNotFoundError("Neither 'refrakt' command nor 'python3'/'python' found in PATH")
            else:
                cmd = [
                    refrakt_cmd,
                    "--config", config_path,
                    "--log-dir", log_dir_relative
                ]
            
            print(f"DEBUG: Running command: {' '.join(cmd)}")
            print(f"DEBUG: Project root: {project_root}")
            print(f"DEBUG: Output dir (absolute): {output_dir}")
            print(f"DEBUG: Log dir (relative): {log_dir_relative}")
            print(f"DEBUG: Config path: {config_path}")
            print(f"DEBUG: Config exists: {os.path.exists(config_path)}")
            
            # Prepare environment variables for tqdm and backend detection
            env = os.environ.copy()
            if 'TERM' not in env:
                env['TERM'] = 'xterm-256color'
            env['TQDM_DISABLE'] = '0'
            
            # Set environment variable to indicate backend execution and job directory
            # Use absolute path for REFRAKT_JOB_DIR
            env['REFRAKT_JOB_DIR'] = str(output_dir.resolve())
            
            # Run the command with real-time output streaming
            # Set cwd to project root so refrakt CLI runs from the correct directory
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=str(project_root),
                    env=env
                )
            except FileNotFoundError as e:
                error_msg = f"Failed to start subprocess: {e}. Command: {' '.join(cmd)}"
                print(f"DEBUG: {error_msg}")
                raise FileNotFoundError(error_msg) from e
            
            # Stream output in real-time using chunked reading
            # This handles both \n and \r sequences from tqdm progress bars
            output_lines = []
            if process.stdout:
                async for line_text in self._read_stream_chunks(process.stdout):
                    if line_text:
                        output_lines.append(line_text)
                        
                        # Store in job logs
                        self.add_log(job_id, line_text)
                        
                        # Broadcast to WebSocket clients
                        await self.websocket_service.broadcast_log(job_id, line_text)
                        
                        print(f"[JOB {job_id}] {line_text}")
            
            # Wait for process to complete
            await process.wait()
            
            # Update job status
            if process.returncode == 0:
                self.update_job_status(job_id, "uploading")
                await self.websocket_service.broadcast_log(
                    job_id,
                    "📦 Uploading artifacts to R2..."
                )
                
                # Upload artifacts to R2
                upload_stats = await self.r2_service.upload_job_artifacts(
                    job_id,
                    output_dir
                )
                
                self.jobs[job_id]["r2_uploaded"] = upload_stats["uploaded"] > 0
                self.jobs[job_id]["r2_stats"] = upload_stats
                
                self.update_job_status(job_id, "completed")
                self.jobs[job_id]["result_path"] = str(output_dir)
                
                await self.websocket_service.broadcast_log(
                    job_id,
                    "✅ Job completed and artifacts uploaded!"
                )
                print(f"DEBUG: Job {job_id} completed successfully")
            else:
                error_msg = "\n".join(output_lines[-10:]) if output_lines else "Unknown error"
                self.update_job_status(job_id, "error", error=error_msg)
                print(f"DEBUG: Job {job_id} failed with return code {process.returncode}")
            
        except Exception as e:
            self.update_job_status(job_id, "error", error=str(e))
            print(f"DEBUG: Job {job_id} failed with exception: {str(e)}")
    
    async def _read_stream_chunks(self, stream, chunk_size: int = 8192):
        """
        Read from stream in chunks, handling both newline and carriage return sequences.
        This function handles tqdm progress bars that use \r for line updates.
        
        Args:
            stream: Async stream to read from
            chunk_size: Size of chunks to read
        
        Yields:
            Decoded text lines or chunks
        """
        buffer = b""
        
        while True:
            try:
                chunk = await stream.read(chunk_size)
                if not chunk:
                    # Process remaining buffer before breaking
                    if buffer:
                        try:
                            remaining = buffer.decode('utf-8', errors='replace').rstrip()
                            if remaining:
                                yield remaining
                        except Exception:
                            pass
                    break
                
                buffer += chunk
                
                # Process buffer looking for line separators
                # Handle both \n (newline) and \r (carriage return) sequences
                processed = True
                while processed:
                    processed = False
                    if b'\n' in buffer:
                        line_bytes, buffer = buffer.split(b'\n', 1)
                        try:
                            line_text = line_bytes.decode('utf-8', errors='replace').rstrip('\r').strip()
                            if line_text:
                                yield line_text
                        except Exception as e:
                            print(f"DEBUG: Decode error: {e}")
                        processed = True
                        continue
                    if b'\r' in buffer:
                        parts = buffer.split(b'\r', 1)
                        if len(parts) == 2:
                            line_bytes, buffer = parts
                            try:
                                line_text = line_bytes.decode('utf-8', errors='replace').strip()
                                if line_text:
                                    yield line_text
                            except Exception:
                                pass
                            processed = True
                            continue
                    if len(buffer) > chunk_size * 4:
                        # Extract a chunk to prevent buffer overflow
                        chunk_to_process = buffer[:chunk_size * 2]
                        buffer = buffer[chunk_size * 2:]
                        try:
                            chunk_text = chunk_to_process.decode('utf-8', errors='replace').rstrip()
                            if chunk_text:
                                yield chunk_text
                        except Exception:
                            pass
                        processed = True
                        continue
                    break
                        
            except Exception as e:
                print(f"DEBUG: Stream read error: {e}")
                break

