"""WebSocket service for real-time log streaming"""

from typing import Dict, List
from fastapi import WebSocket


class WebSocketService:
    """Service for managing WebSocket connections"""
    
    def __init__(self):
        """Initialize WebSocket service"""
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def add_connection(self, job_id: str, websocket: WebSocket):
        """Add a WebSocket connection for a job"""
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)
    
    async def remove_connection(self, job_id: str, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if job_id in self.active_connections:
            if websocket in self.active_connections[job_id]:
                self.active_connections[job_id].remove(websocket)
    
    async def broadcast_log(self, job_id: str, message: str):
        """Broadcast log message to all connected WebSocket clients"""
        if job_id not in self.active_connections:
            return
        
        disconnected = []
        for ws in self.active_connections[job_id]:
            try:
                await ws.send_text(message)
            except Exception:
                disconnected.append(ws)
        
        # Remove disconnected clients
        for ws in disconnected:
            self.active_connections[job_id].remove(ws)
    
    def get_connections_count(self, job_id: str) -> int:
        """Get number of active connections for a job"""
        if job_id not in self.active_connections:
            return 0
        return len(self.active_connections[job_id])

