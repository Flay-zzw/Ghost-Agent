import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# ---------- 加载环境变量 ----------
load_dotenv()

MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")
MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL", "https://api.minimax.io/v1")
MINIMAX_MODEL = os.getenv("MINIMAX_MODEL", "MiniMax-M2.5")


# ---------- 初始化 LLM ----------
# 核心原理：MiniMax 兼容 OpenAI API 格式，
# 所以直接用 ChatOpenAI，把 base_url 指向 MiniMax 就行
def create_llm(streaming: bool = False) -> ChatOpenAI:
    if not MINIMAX_API_KEY:
        raise ValueError("请在 .env 文件中设置 MINIMAX_API_KEY")
    return ChatOpenAI(
        model=MINIMAX_MODEL,
        api_key=MINIMAX_API_KEY,
        base_url=MINIMAX_BASE_URL,
        temperature=0.7,
        max_tokens=2048,
        streaming=streaming,
    )


# ---------- Prompt 模板 ----------
SYSTEM_PROMPT = """你是一个友好、专业的 AI 助手。请用简洁清晰的中文回答用户问题。"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])


# ---------- 会话历史（内存版，生产环境换 Redis） ----------
sessions: dict[str, list] = {}


def get_history(session_id: str) -> list:
    return sessions.setdefault(session_id, [])


def save_turn(session_id: str, human_msg: str, ai_msg: str):
    history = get_history(session_id)
    history.append(HumanMessage(content=human_msg))
    history.append(AIMessage(content=ai_msg))
    # 只保留最近 20 轮对话
    if len(history) > 40:
        sessions[session_id] = history[-40:]


# ---------- FastAPI ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MINIMAX_API_KEY:
        print("⚠️  MINIMAX_API_KEY 未设置，请检查 .env 文件")
    else:
        print(f"✅ 助手已启动 | 模型: {MINIMAX_MODEL} | API: {MINIMAX_BASE_URL}")
    yield


app = FastAPI(title="MiniMax 智能助手", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- 请求 / 响应模型 ----------
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    reply: str
    session_id: str


# ---------- 你原来的路由（保留） ----------
@app.get("/")
async def root():
    return {"message": "Hello World", "service": "MiniMax 智能助手"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


# ---------- 新增：AI 对话路由 ----------
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """普通对话 - 返回完整回复"""
    try:
        llm = create_llm(streaming=False)
        chain = prompt | llm | StrOutputParser()

        result = await chain.ainvoke({
            "history": get_history(req.session_id),
            "input": req.message,
        })

        save_turn(req.session_id, req.message, result)
        return ChatResponse(reply=result, session_id=req.session_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"调用失败: {str(e)}")


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """流式对话 - SSE 逐字输出"""

    async def generate():
        try:
            llm = create_llm(streaming=True)
            chain = prompt | llm | StrOutputParser()
            full_response = ""

            async for chunk in chain.astream({
                "history": get_history(req.session_id),
                "input": req.message,
            }):
                full_response += chunk
                yield f"data: {chunk}\n\n"

            save_turn(req.session_id, req.message, full_response)
            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """清除会话历史"""
    sessions.pop(session_id, None)
    return {"message": f"会话 {session_id} 已清除"}
