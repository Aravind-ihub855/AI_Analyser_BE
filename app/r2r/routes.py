from fastapi import APIRouter, HTTPException, Body, Depends
from app.r2r.schema import QuestionRequest, SummarizeRequest, ChatResponse
from app.r2r.chatbot import summarize_messages
from app.r2r import chat_history
from app.shared.authMiddleware import get_current_user
from app.ai_analyzer.agent_orchestrator import AgentOrchestrator
from app.ai_analyzer.db_sync import sync_mongodb_to_sqlite
import os

router = APIRouter(prefix="/r2r", tags=["Chat"])

orchestrator = AgentOrchestrator()
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ai_analyzer", "db", "store_ops.db")

@router.post("/sync")
async def sync_db(current_user: dict = Depends(get_current_user)):
    try:
        success = await sync_mongodb_to_sqlite(DB_PATH)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to sync remote database")
        return {"message": "Data synchronized successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi.responses import StreamingResponse
import json

@router.post("/chat", response_model=ChatResponse)
async def ask_ai(request: QuestionRequest, current_user: dict = Depends(get_current_user)):
    try:
        # Pass conversation context and current user to the orchestrator
        res = await orchestrator.process(request.question, request.chat_context, current_user)
        
        # Formulate tableData to match expected ChatResponse format
        table_dict = None
        if res.get("tableData") and len(res["tableData"]) > 0:
            table_dict = {
                "isTable": True,
                "headers": res["tableData"][0],
                "rows": [[str(v) if v is not None else "" for v in row] for row in res["tableData"][1:]]
            }
            
        return ChatResponse(
            answer=res.get("response", ""),
            tableData=table_dict,
            chartData=None,
            reportData=res.get("reportData"),
            voice_enabled=False,
            metadata={
                "model": res.get("model_name", "mistral-medium-2505"),
                "token_usage": res.get("token_usage", {}),
                "chartConfig": res.get("chartData")  # JSON chart config for frontend
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/stream")
async def ask_ai_stream(request: QuestionRequest, current_user: dict = Depends(get_current_user)):
    """Streams token chunks and metadata logs via Server-Sent Events (SSE)."""
    async def sse_generator():
        try:
            async for chunk in orchestrator.process_stream(request.question, request.chat_context, current_user):
                yield f"data: {chunk}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            
    return StreamingResponse(sse_generator(), media_type="text/event-stream")

@router.get("/chats")
async def list_user_chats(current_user: dict = Depends(get_current_user)):
    try:
        user_id = current_user["id"]
        chats = await chat_history.get_user_chats(user_id)
        return {"chats": chats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chats/new")
async def create_chat(current_user: dict = Depends(get_current_user)):
    try:
        user_id = current_user["id"]
        chat_id = await chat_history.create_new_chat(user_id)
        return {"chat_id": chat_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chats/{chat_id}")
async def get_history(chat_id: str, current_user: dict = Depends(get_current_user)):
    try:
        details = await chat_history.get_chat_details(chat_id)
        if not details:
            raise HTTPException(status_code=404, detail="Chat not found")
        # Verify ownership
        if details.get("user_id") != current_user["id"]:
            raise HTTPException(status_code=403, detail="Forbidden: You do not own this chat")
        return details
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/chats/{chat_id}/title")
async def rename_chat(chat_id: str, title: str = Body(..., embed=True), current_user: dict = Depends(get_current_user)):
    try:
        details = await chat_history.get_chat_details(chat_id)
        if not details:
            raise HTTPException(status_code=404, detail="Chat not found")
        if details.get("user_id") != current_user["id"]:
            raise HTTPException(status_code=403, detail="Forbidden")
        await chat_history.update_chat_title(chat_id, title)
        return {"status": "success"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str, current_user: dict = Depends(get_current_user)):
    try:
        details = await chat_history.get_chat_details(chat_id)
        if not details:
            raise HTTPException(status_code=404, detail="Chat not found")
        if details.get("user_id") != current_user["id"]:
            raise HTTPException(status_code=403, detail="Forbidden")
        await chat_history.delete_chat_session(chat_id)
        return {"status": "success"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chats/message")
async def add_message(payload: dict = Body(...), current_user: dict = Depends(get_current_user)):
    try:
        chat_id = payload.get("chat_id")
        role = payload.get("role")
        content = payload.get("content")
        tableData = payload.get("tableData")
        chartData = payload.get("chartData")
        reportData = payload.get("reportData")
        dashboardData = payload.get("dashboardData")
        
        details = await chat_history.get_chat_details(chat_id)
        if not details:
            raise HTTPException(status_code=404, detail="Chat not found")
        if details.get("user_id") != current_user["id"]:
            raise HTTPException(status_code=403, detail="Forbidden")
            
        await chat_history.save_chat_message(
            chat_id, role, content, 
            tableData, chartData, reportData, dashboardData
        )
        return {"status": "success"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/summarize")
async def summarize_chat(request: SummarizeRequest, current_user: dict = Depends(get_current_user)):
    try:
        summary = await summarize_messages(request.messages)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

