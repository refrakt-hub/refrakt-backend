"""Job management service"""

import asyncio
import os
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from config import get_settings
from services.r2_service import R2Service
from services.websocket_service import WebSocketService


class JobService:
    """Service for managing ML training jobs"""
    
    def __init__(self, r2_service: R2Service, websocket_service: WebSocketService):
        """Initialize job service"""
        self.settings = get_settings()
        self.r2_service = r2_service
        self.websocket_service = websocket_service
        self.jobs: Dict[str, dict] = {}
    
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
            "r2_uploaded": False
        }
        
        return job_id
    
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
    
    def cleanup_job_directory(self, job_id: str, job_dir: Path) -> bool:
        """
        Clean up local job directory after successful R2 upload
        
        Args:
            job_id: Job identifier
            job_dir: Job directory path to clean up
            
        Returns:
            True if cleanup was successful, False otherwise
        """
        try:
            if not job_dir.exists():
                print(f"Job directory already cleaned up: {job_dir}")
                return True
            
            # Delete the entire job directory
            shutil.rmtree(job_dir)
            print(f"Cleaned up local files for job {job_id}")
            return True
        except Exception as e:
            print(f"Error cleaning up job directory {job_dir}: {str(e)}")
            return False
    
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
            cmd = [
                "refrakt",
                "--config", config_path,
                "--log-dir", str(output_dir)
            ]
            
            print(f"DEBUG: Running command: {' '.join(cmd)}")
            
            # Run the command with real-time output streaming
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=os.getcwd()
            )
            
            # Stream output in real-time
            output_lines = []
            if process.stdout:
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        break
                    
                    line_text = line.decode().rstrip()
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
                    "Uploading artifacts to R2..."
                )
                
                # Upload artifacts to R2
                upload_stats = await self.r2_service.upload_job_artifacts(
                    job_id,
                    output_dir
                )
                
                self.jobs[job_id]["r2_uploaded"] = upload_stats["uploaded"] > 0
                self.jobs[job_id]["r2_stats"] = upload_stats
                
                # Store artifact metadata before cleanup (for listing after cleanup)
                artifact_metadata = []
                if output_dir.exists():
                    for file_path in output_dir.rglob('*'):
                        if file_path.is_file():
                            try:
                                stat = file_path.stat()
                                relative_path = str(file_path.relative_to(output_dir))
                                artifact_metadata.append({
                                    "name": file_path.name,
                                    "path": relative_path,
                                    "size": stat.st_size,
                                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                                })
                            except Exception as e:
                                print(f"Error collecting artifact metadata for {file_path}: {str(e)}")
                
                self.jobs[job_id]["artifact_metadata"] = artifact_metadata
                
                # Clean up local files if upload was successful
                # Only cleanup if all files were uploaded successfully (no failures)
                if upload_stats["uploaded"] > 0 and upload_stats["failed"] == 0:
                    await self.websocket_service.broadcast_log(
                        job_id,
                        "Cleaning up local files..."
                    )
                    cleanup_success = self.cleanup_job_directory(job_id, output_dir)
                    if cleanup_success:
                        await self.websocket_service.broadcast_log(
                            job_id,
                            "Local files cleaned up successfully"
                        )
                        self.jobs[job_id]["local_cleaned"] = True
                    else:
                        await self.websocket_service.broadcast_log(
                            job_id,
                            "Warning: Failed to clean up local files"
                        )
                elif upload_stats["failed"] > 0:
                    await self.websocket_service.broadcast_log(
                        job_id,
                        f"Keeping local files due to {upload_stats['failed']} upload failure(s)"
                    )
                
                self.update_job_status(job_id, "completed")
                self.jobs[job_id]["result_path"] = str(output_dir)
                
                await self.websocket_service.broadcast_log(
                    job_id,
                    "Job completed and artifacts uploaded!"
                )
                print(f"DEBUG: Job {job_id} completed successfully")
            else:
                error_msg = "\n".join(output_lines[-10:]) if output_lines else "Unknown error"
                self.update_job_status(job_id, "error", error=error_msg)
                print(f"DEBUG: Job {job_id} failed with return code {process.returncode}")
            
        except Exception as e:
            self.update_job_status(job_id, "error", error=str(e))
            print(f"DEBUG: Job {job_id} failed with exception: {str(e)}")

