"""Cloudflare R2 storage service"""

import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from config import get_settings


class R2Service:
    """Service for interacting with Cloudflare R2 storage"""
    
    def __init__(self):
        """Initialize R2 client"""
        settings = get_settings()
        self.settings = settings
        
        if settings.is_r2_configured:
            self.s3_client = boto3.client(
                's3',
                endpoint_url=settings.r2_endpoint_url,
                aws_access_key_id=settings.R2_ACCESS_KEY_ID,
                aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
                config=Config(signature_version='s3v4'),
                region_name='auto'
            )
            print("Cloudflare R2 configured successfully")
        else:
            self.s3_client = None
            print("R2 credentials not found - artifact upload will be disabled")
    
    def is_configured(self) -> bool:
        """Check if R2 is configured"""
        return self.s3_client is not None
    
    async def upload_file(
        self,
        job_id: str,
        local_path: Path,
        relative_path: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Upload a file to Cloudflare R2
        
        Args:
            job_id: Job identifier
            local_path: Local file path
            relative_path: Relative path within job directory
            
        Returns:
            Tuple of (success, public_url)
        """
        if not self.s3_client:
            return False, None
        
        try:
            # Generate R2 key (path in bucket)
            r2_key = f"jobs/{job_id}/{relative_path}"
            
            # Determine content type
            content_type, _ = mimetypes.guess_type(str(local_path))
            if content_type is None:
                content_type = "application/octet-stream"
            
            # Upload file
            with open(local_path, 'rb') as file_data:
                self.s3_client.upload_fileobj(
                    file_data,
                    self.settings.R2_BUCKET_NAME,
                    r2_key,
                    ExtraArgs={
                        'ContentType': content_type,
                        'Metadata': {
                            'job_id': job_id,
                            'uploaded_at': datetime.now().isoformat()
                        }
                    }
                )
            
            # Generate public URL
            public_url = f"{self.settings.R2_PUBLIC_URL}/{r2_key}"
            
            print(f"Uploaded to R2: {r2_key}")
            return True, public_url
            
        except ClientError as e:
            print(f"R2 upload error: {str(e)}")
            return False, None
        except Exception as e:
            print(f"Unexpected upload error: {str(e)}")
            return False, None
    
    def generate_presigned_url(
        self,
        job_id: str,
        relative_path: str,
        expiration: int = 3600
    ) -> Optional[str]:
        """
        Generate a presigned URL for downloading from R2
        
        Args:
            job_id: Job identifier
            relative_path: Relative path within job directory
            expiration: URL expiration time in seconds (default: 1 hour)
            
        Returns:
            Presigned URL or None if failed
        """
        if not self.s3_client:
            return None
        
        try:
            r2_key = f"jobs/{job_id}/{relative_path}"
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.settings.R2_BUCKET_NAME,
                    'Key': r2_key
                },
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            print(f"Error generating presigned URL: {str(e)}")
            return None
    
    async def upload_job_artifacts(self, job_id: str, job_dir: Path) -> dict:
        """
        Upload all job artifacts to R2
        
        Args:
            job_id: Job identifier
            job_dir: Job directory path
            
        Returns:
            Dictionary with upload statistics
        """
        if not self.s3_client:
            print(f"Skipping R2 upload for job {job_id} - R2 not configured")
            return {"uploaded": 0, "failed": 0, "total": 0}
        
        if not job_dir.exists():
            print(f"Job directory not found: {job_dir}")
            return {"uploaded": 0, "failed": 0, "total": 0}
        
        uploaded_count = 0
        failed_count = 0
        
        # Upload all files
        for file_path in job_dir.rglob('*'):
            if file_path.is_file():
                relative_path = str(file_path.relative_to(job_dir))
                success, _ = await self.upload_file(job_id, file_path, relative_path)
                
                if success:
                    uploaded_count += 1
                else:
                    failed_count += 1
        
        print(f"R2 Upload Summary for {job_id}: {uploaded_count} uploaded, {failed_count} failed")
        
        return {
            "uploaded": uploaded_count,
            "failed": failed_count,
            "total": uploaded_count + failed_count
        }
    
    def test_connection(self) -> dict:
        """Test R2 connection and permissions"""
        if not self.s3_client:
            return {
                "status": "error",
                "message": "R2 not configured",
                "configured": False
            }
        
        try:
            # Try to list objects (limited to 1)
            response = self.s3_client.list_objects_v2(
                Bucket=self.settings.R2_BUCKET_NAME,
                MaxKeys=1
            )
            
            return {
                "status": "success",
                "message": "R2 connection successful",
                "configured": True,
                "bucket": self.settings.R2_BUCKET_NAME,
                "can_list": True
            }
        except ClientError as e:
            return {
                "status": "error",
                "message": f"R2 connection failed: {str(e)}",
                "configured": True,
                "error": str(e)
            }

