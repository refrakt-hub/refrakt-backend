#!/usr/bin/env python3
"""
Refrakt Backend API Server - Enhanced with Cloudflare R2 Storage

Features:
1. Upload job artifacts to Cloudflare R2
2. Generate presigned URLs for secure downloads
3. Real-time log streaming via WebSocket
4. Job artifacts management
"""

import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import router
from config import get_settings

# Initialize settings
settings = get_settings()

# Initialize FastAPI app
app = FastAPI(
    title="Refrakt Backend API",
    description="Backend API for Refrakt ML Framework with R2 Storage",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        env = sys.argv[1]
        if env == "dev":
            port = 8001
            reload = True
            print(f"Starting Refrakt Backend in DEVELOPMENT mode on port {port}")
            print(f"Access via: http://localhost:{port}")
            print(f"Hot reload enabled for development")
        elif env == "prod":
            port = 8002
            reload = False
            print(f"Starting Refrakt Backend in PRODUCTION mode on port {port}")
        else:
            port = 8000
            reload = False
    else:
        port = 8000
        reload = False
        print(f"Usage: python main.py [dev|prod]")
    
    if reload:
        uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
    else:
        uvicorn.run(app, host="0.0.0.0", port=port)
