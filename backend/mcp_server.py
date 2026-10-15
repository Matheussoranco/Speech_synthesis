"""
MCP (Model Context Protocol) Server for VoiceStudio
Allows AI agents to use voice synthesis capabilities
"""

import json
import asyncio
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

from core.config import settings

# MCP Protocol types
class MCPRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: Optional[str] = None
    method: str
    params: Optional[Dict[str, Any]] = None

class MCPResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

class Tool(BaseModel):
    name: str
    description: str
    inputSchema: Dict[str, Any]

class TextContent(BaseModel):
    type: str = "text"
    text: str

class CallToolResult(BaseModel):
    content: List[TextContent]
    isError: bool = False

# Available tools
TOOLS = [
    Tool(
        name="synthesize_speech",
        description="Generate speech from text using TTS",
        inputSchema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to synthesize"},
                "voice": {"type": "string", "description": "Voice to use (default: default)"},
                "language": {"type": "string", "description": "Language code (default: en)"},
                "speed": {"type": "number", "description": "Speech speed (default: 1.0)"},
            },
            "required": ["text"],
        },
    ),
    Tool(
        name="clone_voice",
        description="Clone a voice from reference audio and synthesize speech",
        inputSchema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to synthesize"},
                "reference_audio_base64": {"type": "string", "description": "Reference audio as base64"},
                "reference_audio_path": {"type": "string", "description": "Path to reference audio file"},
                "language": {"type": "string", "description": "Language code (default: auto)"},
                "profile_id": {"type": "string", "description": "Saved voice profile ID"},
            },
            "required": ["text"],
        },
    ),
    Tool(
        name="list_voices",
        description="List available voices and engines",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="transcribe_audio",
        description="Transcribe audio to text with optional speaker diarization",
        inputSchema={
            "type": "object",
            "properties": {
                "audio_base64": {"type": "string", "description": "Audio as base64"},
                "audio_path": {"type": "string", "description": "Path to audio file"},
                "language": {"type": "string", "description": "Language hint (default: auto)"},
                "diarize": {"type": "boolean", "description": "Enable speaker diarization (default: false)"},
            },
            "required": [],
        },
    ),
    Tool(
        name="create_voice_profile",
        description="Save a voice profile for reuse",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Profile name"},
                "reference_audio_base64": {"type": "string", "description": "Reference audio as base64"},
                "reference_text": {"type": "string", "description": "What the speaker says in the reference"},
                "language": {"type": "string", "description": "Language code"},
            },
            "required": ["name", "reference_audio_base64", "reference_text", "language"],
        },
    ),
    Tool(
        name="list_voice_profiles",
        description="List saved voice profiles",
        inputSchema={"type": "object", "properties": {}},
    ),
]

# Tool handlers
async def handle_synthesize_speech(params: Dict[str, Any]) -> CallToolResult:
    """Handle TTS synthesis"""
    # In production, call the actual generate endpoint
    return CallToolResult(
        content=[TextContent(text=f"Would synthesize: {params.get('text', '')[:50]}...")],
        isError=False,
    )

async def handle_clone_voice(params: Dict[str, Any]) -> CallToolResult:
    """Handle voice cloning"""
    return CallToolResult(
        content=[TextContent(text=f"Would clone voice for: {params.get('text', '')[:50]}...")],
        isError=False,
    )

async def handle_list_voices(params: Dict[str, Any]) -> CallToolResult:
    """List available voices"""
    from api.routers.engines import get_engines
    engines = await get_engines()
    voices = []
    for engine in engines.tts:
        if engine.status == "available":
            voices.append(f"{engine.id}: {engine.name}")
    return CallToolResult(
        content=[TextContent(text="Available voices:\n" + "\n".join(voices) if voices else "No voices available")],
        isError=False,
    )

async def handle_transcribe_audio(params: Dict[str, Any]) -> CallToolResult:
    """Handle transcription"""
    return CallToolResult(
        content=[TextContent(text="Transcription not yet implemented via MCP")],
        isError=False,
    )

async def handle_create_voice_profile(params: Dict[str, Any]) -> CallToolResult:
    """Create voice profile"""
    return CallToolResult(
        content=[TextContent(text=f"Would create profile: {params.get('name')}")],
        isError=False,
    )

async def handle_list_voice_profiles(params: Dict[str, Any]) -> CallToolResult:
    """List voice profiles"""
    from api.routers.profiles import list_profiles
    profiles = await list_profiles()
    return CallToolResult(
        content=[TextContent(text="Saved profiles:\n" + "\n".join(f"{p.id}: {p.name}" for p in profiles) if profiles else "No profiles")],
        isError=False,
    )

TOOL_HANDLERS = {
    "synthesize_speech": handle_synthesize_speech,
    "clone_voice": handle_clone_voice,
    "list_voices": handle_list_voices,
    "transcribe_audio": handle_transcribe_audio,
    "create_voice_profile": handle_create_voice_profile,
    "list_voice_profiles": handle_list_voice_profiles,
}

# MCP Server
class MCPServer:
    def __init__(self):
        self.tools = TOOLS
        self.handlers = TOOL_HANDLERS
    
    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        try:
            if request.method == "initialize":
                return MCPResponse(
                    id=request.id,
                    result={
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "voicestudio", "version": settings.app_version},
                    },
                )
            
            elif request.method == "tools/list":
                return MCPResponse(
                    id=request.id,
                    result={"tools": [t.model_dump() for t in self.tools]},
                )
            
            elif request.method == "tools/call":
                params = request.params or {}
                tool_name = params.get("name")
                tool_args = params.get("arguments", {})
                
                if tool_name not in self.handlers:
                    return MCPResponse(
                        id=request.id,
                        error={"code": -32601, "message": f"Unknown tool: {tool_name}"},
                    )
                
                result = await self.handlers[tool_name](tool_args)
                return MCPResponse(
                    id=request.id,
                    result=result.model_dump(),
                )
            
            else:
                return MCPResponse(
                    id=request.id,
                    error={"code": -32601, "message": f"Method not found: {request.method}"},
                )
        
        except Exception as e:
            return MCPResponse(
                id=request.id,
                error={"code": -32603, "message": str(e)},
            )

mcp_server = MCPServer()

# FastAPI app for MCP (can be mounted or run standalone)
mcp_app = FastAPI(title="VoiceStudio MCP Server")

@mcp_app.post("/mcp")
async def mcp_endpoint(request: Request):
    body = await request.json()
    mcp_request = MCPRequest(**body)
    response = await mcp_server.handle_request(mcp_request)
    return response.model_dump(exclude_none=True)

@mcp_app.get("/mcp")
async def mcp_sse():
    """SSE endpoint for MCP (for clients that prefer SSE)"""
    async def event_stream():
        yield 'data: {"jsonrpc":"2.0","method":"notifications/initialized"}\n\n'
        # Keep alive
        while True:
            await asyncio.sleep(30)
            yield ": keepalive\n\n"
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )

# Standalone runner
if __name__ == "__main__":
    uvicorn.run(mcp_app, host="127.0.0.1", port=3901)
