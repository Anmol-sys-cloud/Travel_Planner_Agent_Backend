from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from utils.save_to_document import save_document
from starlette.responses import JSONResponse
import os
import datetime
from dotenv import load_dotenv
from pydantic import BaseModel
load_dotenv()
from langchain_core.messages import HumanMessage, AIMessage
import uuid
from typing import Optional
from utils.json_store import JsonStore

app = FastAPI()

# simple JSON-backed chat store (used by the Streamlit UI)
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
os.makedirs(data_dir, exist_ok=True)
store = JsonStore(os.path.join(data_dir, "chats.json"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # set specific origins in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class QueryRequest(BaseModel):
    question: str
    model: Optional[str] = None


class ChatCreate(BaseModel):
    title: Optional[str] = "New Chat"
    model: Optional[str] = None


class MessageCreate(BaseModel):
    role: str
    content: str

@app.post("/query")
async def query_travel_agent(query:QueryRequest):
    try:
        print(query)
        # Import heavy agent builder here to avoid import-time side-effects
        # (helps when running uvicorn --reload on Windows which uses subprocesses)
        from agent.agentic_workflow import GraphBuilder

        model_choice = query.model or "groq"
        graph = GraphBuilder(model_choice=model_choice)
        react_app=graph()
        #react_app = graph.build_graph()

        png_graph = react_app.get_graph().draw_mermaid_png()
        with open("my_graph.png", "wb") as f:
            f.write(png_graph)

        print(f"Graph saved as 'my_graph.png' in {os.getcwd()}")
        # Build structured messages expected by the LLM/graph runtime.
        # Include an explicit, single-shot instruction to produce a detailed plan.
        detailed_instructions = (
            "Please produce a complete, comprehensive travel plan in Markdown. Include a day-by-day itinerary, "
            "recommended hotels with approximate per-night costs, places of attraction, recommended restaurants with "
            "price ranges, activities, transport options, a detailed cost breakdown, per-day budget, and weather. "
            "Provide two variants if possible: a standard tourist plan and an off-beat plan."
        )
        human = HumanMessage(content=f"{query.question}\n\n{detailed_instructions}")
        messages = {"messages": [human]}

        # determine a safe log path next to this file
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        except Exception:
            base_dir = os.getcwd()
        raw_path = os.path.join(base_dir, "last_raw_output.txt")
        err_path = os.path.join(base_dir, "last_error.txt")

        # Invoke the graph runtime and capture any exceptions/outputs to log files
        try:
            output = react_app.invoke(messages)
            try:
                with open(raw_path, "w", encoding="utf-8") as lof:
                    lof.write(repr(output))
            except Exception:
                pass
        except Exception as invoke_exc:
            import traceback
            trace = traceback.format_exc()
            try:
                with open(err_path, "w", encoding="utf-8") as ef:
                    ef.write(trace)
            except Exception:
                pass
            return JSONResponse(status_code=500, content={"error": "invoke_failed", "trace": trace})

        # Robust extraction of text from common response shapes
        final_output = None
        ai_last_message = None
        if isinstance(output, dict):
            if "messages" in output and output["messages"]:
                ai_last_message = output["messages"][-1]
                final_output = getattr(ai_last_message, "content", None) or str(ai_last_message)
            elif "content" in output:
                final_output = output["content"]
            else:
                final_output = str(output)
        elif hasattr(output, "content"):
            ai_last_message = output
            final_output = output.content
        else:
            final_output = str(output)


            #  attached the code till here..

        # If the model returned a function-call failure that embeds a 'failed_generation'
        # or a partial plan inside the error text, try to extract the Markdown plan
        try:
            fo_lower = (final_output or "").lower()
            if "failed to call a function" in fo_lower or "failed_generation" in fo_lower or (
                isinstance(final_output, str) and final_output.startswith("Error: Error code:")
            ):
                import re

                # Look for an embedded markdown plan starting with a top-level header
                m = re.search(r"(#\s+[^\n].*)", final_output, flags=re.DOTALL)
                if m:
                    final_output = m.group(1).strip()
                else:
                    # As a fallback, try to find 'failed_generation' marker and grab what's after it
                    idx = fo_lower.find("failed_generation")
                    if idx != -1:
                        # find the first newline after the marker in the original string
                        orig = final_output
                        start = orig.lower().find("failed_generation")
                        # attempt to skip to the next newline and take the remainder
                        nl = orig.find("\n", start)
                        if nl != -1:
                            final_output = orig[nl + 1 :].strip()
                        else:
                            # last resort: keep original final_output but remove the Error prefix
                            parts = orig.split("failed_generation", 1)
                            final_output = parts[-1].strip()
        except Exception:
            pass

        # Log raw output for debugging
        try:
            with open("last_raw_output.txt", "w", encoding="utf-8") as lof:
                lof.write(repr(output))
        except Exception:
            pass

        return {"answer": final_output}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/chats")
async def create_chat(chat: ChatCreate):
    chat_id = uuid.uuid4().hex
    now = datetime.datetime.utcnow().isoformat()
    chat_obj = {
        "id": chat_id,
        "title": chat.title or "New Chat",
        "model": chat.model,
        "created_at": now,
        "updated_at": now,
        "pinned": False,
        "messages": [],
    }
    store.save_chat(chat_obj)
    return chat_obj


@app.get("/chats")
async def list_chats():
    return store.list_chats()


@app.get("/chats/{chat_id}")
async def get_chat(chat_id: str):
    chat = store.get_chat(chat_id)
    if not chat:
        return JSONResponse(status_code=404, content={"error": "not_found"})
    return chat


@app.post("/chats/{chat_id}/messages")
async def append_message(chat_id: str, message: MessageCreate):
    chat = store.get_chat(chat_id)
    if not chat:
        return JSONResponse(status_code=404, content={"error": "not_found"})

    now = datetime.datetime.utcnow().isoformat()
    msg = {"role": message.role, "content": message.content, "ts": now}
    store.append_message(chat_id, msg)

    # Simple placeholder assistant reply. Replace with real LLM/agent call later.
    assistant_text = f"Assistant (mock): I received your message: {message.content}"
    assistant_msg = {"role": "assistant", "content": assistant_text, "ts": datetime.datetime.utcnow().isoformat()}
    store.append_message(chat_id, assistant_msg)

    return assistant_msg


@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str):
    ok = store.delete_chat(chat_id)
    if not ok:
        return JSONResponse(status_code=404, content={"error": "not_found"})
    return {"deleted": chat_id}


@app.get("/chats/search")
async def search_chats(q: Optional[str] = None):
    if not q:
        return store.list_chats()
    return store.search_chats(q)