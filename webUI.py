"""Session-only Qwen RAG demo built with Gradio and LangGraph.

Access state, API credentials, document vectors, and conversations live only
in Gradio session state. Refreshing the page starts an empty session.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import mimetypes
import os
import re
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Sequence, TypedDict

import gradio as gr
import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

try:
    from pypdf import PdfReader
except ImportError:  # Shows a clear UI error if dependencies are incomplete.
    PdfReader = None


load_dotenv()
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("project2.web")

QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
CHAT_MODEL = "qwen-plus"
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_DIMENSIONS = 1024

SESSION_TTL_SECONDS = 30 * 60
DEFAULT_TITLE = "新对话"
MAX_CONVERSATIONS = 20
MAX_DOCUMENTS = 3
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 50
MAX_DOCUMENT_CHARS = 100_000
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 5

TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""],
)


class RagState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: str


class DocumentError(ValueError):
    """A document error safe to show in the UI."""


def _release_state(value: Any) -> None:
    if isinstance(value, (dict, list)):
        value.clear()


def _status(message: str = "", kind: str = "info", visible: bool | None = None):
    if visible is None:
        visible = bool(message)
    icons = {"success": "✓", "error": "!", "info": "·"}
    value = f"{icons.get(kind, '·')} {message}" if message else ""
    return gr.update(value=value, visible=visible)


def _friendly_provider_error(exc: Exception, operation: str) -> str:
    """Map provider failures without logging bodies, headers, or credentials."""
    detail = str(exc).lower()
    logger.warning("Qwen %s failed: %s", operation, type(exc).__name__)
    if any(marker in detail for marker in ("401", "invalid_api_key", "incorrect api key", "unauthorized")):
        return "Qwen API Key 无效，或不是阿里云百炼北京地域的 Key。"
    if any(marker in detail for marker in ("429", "rate limit", "throttl")):
        return "Qwen 请求过于频繁，请稍后再试。"
    if any(marker in detail for marker in ("arrearage", "insufficient", "quota", "balance")):
        return "Qwen 账户额度不足，请检查百炼控制台。"
    if any(marker in detail for marker in ("timeout", "timed out", "connect")):
        return "连接 Qwen 服务超时，请检查网络后重试。"
    return f"{operation}失败，请稍后重试并检查 Qwen Key 与模型权限。"


def _chat_model(api_key: str, *, validate_only: bool = False) -> ChatOpenAI:
    options: dict[str, Any] = {
        "base_url": QWEN_BASE_URL,
        "api_key": api_key,
        "model": CHAT_MODEL,
        "temperature": 0.2,
        "timeout": (10, 180),
        "max_retries": 2,
        "streaming": not validate_only,
    }
    return ChatOpenAI(**options)


def _embedding_model(api_key: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        base_url=QWEN_BASE_URL,
        api_key=api_key,
        model=EMBEDDING_MODEL,
        deployment=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSIONS,
        chunk_size=10,
        max_retries=2,
        timeout=120,
        tiktoken_enabled=False,
        check_embedding_ctx_length=False,
    )


def _new_conversation() -> tuple[str, dict[str, Any]]:
    conversation_id = uuid.uuid4().hex
    return conversation_id, {"title": DEFAULT_TITLE, "created_at": time.time(), "messages": []}


def _initial_conversations() -> tuple[dict[str, Any], str]:
    conversation_id, conversation = _new_conversation()
    return {conversation_id: conversation}, conversation_id


def _conversation_choices(conversations: dict[str, Any] | None) -> list[tuple[str, str]]:
    ordered = sorted(
        (conversations or {}).items(),
        key=lambda item: item[1].get("created_at", 0),
        reverse=True,
    )
    choices: list[tuple[str, str]] = []
    for conversation_id, conversation in ordered:
        title = conversation.get("title") or DEFAULT_TITLE
        display_title = title if len(title) <= 22 else f"{title[:22]}…"
        clock = datetime.fromtimestamp(conversation.get("created_at", time.time())).strftime("%H:%M")
        choices.append((f"{display_title}  ·  {clock}", conversation_id))
    return choices


def _conversation_update(conversations: dict[str, Any], conversation_id: str | None):
    return gr.update(choices=_conversation_choices(conversations), value=conversation_id)


def _document_choices(documents: list[dict[str, Any]] | None) -> list[tuple[str, str]]:
    return [
        (f"{doc['filename']}  ·  {len(doc['chunks'])} 个片段", doc["id"])
        for doc in documents or []
    ]


def _document_update(documents: list[dict[str, Any]], value: str | None = None):
    return gr.update(choices=_document_choices(documents), value=value)


def _format_response(text: str) -> str:
    formatted = re.sub(r"<think>", "<details><summary>思考过程</summary>\n\n", text)
    formatted = re.sub(r"</think>", "\n\n</details>\n\n", formatted)
    return formatted.strip()


def _normalize_rows(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise DocumentError("Qwen 未返回有效向量。")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise DocumentError("Qwen 返回了无效的零向量。")
    return matrix / norms


def _rank_chunks(query_vector: Sequence[float], documents: list[dict[str, Any]], top_k: int = TOP_K):
    query = np.asarray(query_vector, dtype=np.float32)
    norm = float(np.linalg.norm(query))
    if not norm:
        return []
    query /= norm
    ranked: list[tuple[float, dict[str, Any]]] = []
    for document in documents:
        vectors = document.get("vectors")
        chunks = document.get("chunks", [])
        if not isinstance(vectors, np.ndarray) or len(chunks) != len(vectors):
            continue
        scores = vectors @ query
        ranked.extend((float(score), chunk) for score, chunk in zip(scores, chunks))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[:top_k]


def _format_context(ranked_chunks: list[tuple[float, dict[str, Any]]]) -> str:
    if not ranked_chunks:
        return "当前会话没有可检索的文档资料。"
    parts = []
    for index, (score, chunk) in enumerate(ranked_chunks, start=1):
        page = f"第 {chunk['page']} 页" if chunk.get("page") else "TXT 文档"
        parts.append(
            f"[资料 {index}] 文件：{chunk['filename']}；位置：{page}；相关度：{score:.3f}\n"
            f"{chunk['content']}"
        )
    return "\n\n".join(parts)


def _extract_pdf(path: Path, filename: str) -> list[dict[str, Any]]:
    if PdfReader is None:
        raise DocumentError("服务器缺少 PyPDF，请先执行 pip install -r requirements.txt。")
    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise DocumentError("PDF 无法读取，文件可能已损坏。") from exc
    if reader.is_encrypted:
        try:
            if reader.decrypt("") == 0:
                raise DocumentError("暂不支持有密码的 PDF。")
        except DocumentError:
            raise
        except Exception as exc:
            raise DocumentError("暂不支持有密码的 PDF。") from exc
    if len(reader.pages) > MAX_PDF_PAGES:
        raise DocumentError(f"PDF 最多支持 {MAX_PDF_PAGES} 页。")

    chunks: list[dict[str, Any]] = []
    total_chars = 0
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = (page.extract_text() or "").strip()
        except Exception as exc:
            raise DocumentError(f"PDF 第 {page_number} 页解析失败。") from exc
        if not page_text:
            continue
        remaining = MAX_DOCUMENT_CHARS - total_chars
        if remaining <= 0:
            break
        page_text = page_text[:remaining]
        total_chars += len(page_text)
        for piece in TEXT_SPLITTER.split_text(page_text):
            if piece.strip():
                chunks.append({"filename": filename, "page": page_number, "content": piece.strip()})
    if total_chars < 20 or not chunks:
        raise DocumentError("没有提取到可用文字；扫描版 PDF 暂不支持，请先进行 OCR。")
    return chunks


def _extract_txt(path: Path, filename: str) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentError("TXT 必须使用 UTF-8 或 UTF-8-SIG 编码。") from exc
    except OSError as exc:
        raise DocumentError("TXT 文件无法读取。") from exc
    text = text.strip()
    if len(text) < 20:
        raise DocumentError("TXT 中没有足够的可用文字。")
    text = text[:MAX_DOCUMENT_CHARS]
    return [
        {"filename": filename, "page": None, "content": piece.strip()}
        for piece in TEXT_SPLITTER.split_text(text)
        if piece.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _uploaded_path(file_value: Any) -> tuple[Path, str]:
    raw_path = getattr(file_value, "name", None) or str(file_value)
    path = Path(raw_path)
    original_name = getattr(file_value, "orig_name", None) or path.name
    return path, Path(original_name).name


def _remove_gradio_temp_file(path: Path) -> None:
    """Delete only browser-uploaded copies under the operating-system temp tree."""
    try:
        resolved = path.resolve()
        temp_root = Path(tempfile.gettempdir()).resolve()
        if resolved.is_relative_to(temp_root) and resolved.is_file():
            resolved.unlink(missing_ok=True)
    except OSError:
        logger.info("Temporary upload cleanup was deferred")


def _prepare_document(file_value: Any, api_key: str) -> dict[str, Any]:
    path, filename = _uploaded_path(file_value)
    if not path.is_file():
        raise DocumentError("上传文件不存在，请重新选择。")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise DocumentError("单个文档不能超过 10 MB。")
    suffix = path.suffix.lower()
    if suffix not in {".pdf", ".txt"}:
        raise DocumentError("只支持 PDF 和 TXT 文件。")
    mime_type = mimetypes.guess_type(filename)[0]
    if mime_type and suffix == ".pdf" and mime_type != "application/pdf":
        raise DocumentError("文件扩展名与 PDF 类型不匹配。")

    fingerprint = _sha256(path)
    chunks = _extract_pdf(path, filename) if suffix == ".pdf" else _extract_txt(path, filename)
    try:
        embeddings = _embedding_model(api_key).embed_documents([chunk["content"] for chunk in chunks])
        vectors = _normalize_rows(embeddings)
    except DocumentError:
        raise
    except Exception as exc:
        raise DocumentError(_friendly_provider_error(exc, "文档向量化")) from exc
    if len(vectors) != len(chunks):
        raise DocumentError("Qwen 返回的向量数量与文档片段不一致。")
    return {
        "id": uuid.uuid4().hex,
        "filename": filename,
        "sha256": fingerprint,
        "chunks": chunks,
        "vectors": vectors,
        "created_at": time.time(),
    }


def _retrieve_context(query: str, documents: list[dict[str, Any]], api_key: str) -> str:
    if not documents:
        return "当前会话尚未上传文档。回答一般性问题时请明确说明未使用用户文档。"
    query_vector = _embedding_model(api_key).embed_query(query)
    return _format_context(_rank_chunks(query_vector, documents))


def _build_rag_graph(api_key: str, documents: list[dict[str, Any]]):
    @tool("retrieve_session_documents")
    def retrieve_session_documents(query: str) -> str:
        """Retrieve passages only from documents in this browser session."""
        return _retrieve_context(query, documents, api_key)

    def retrieve_node(state: RagState):
        last_user = next(
            (message.content for message in reversed(state["messages"]) if isinstance(message, HumanMessage)),
            "",
        )
        return {"context": retrieve_session_documents.invoke({"query": str(last_user)})}

    model = _chat_model(api_key)

    def generate_node(state: RagState):
        system_prompt = (
            "你是一个严谨、友好的中文智能文档助手。先使用给出的会话内资料回答。"
            "若资料中没有答案，应明确说‘上传的资料中没有找到该信息’，不要编造。"
            "对于与文档无关的一般问题可以使用常识回答，但要注明未使用用户文档。"
            "引用资料时，在回答末尾增加‘参考来源’，列出文件名和页码；TXT 不写页码。\n\n"
            f"当前检索资料：\n{state.get('context', '')}"
        )
        history = list(state["messages"])[-20:]
        return {"messages": [model.invoke([SystemMessage(content=system_prompt), *history])]}

    workflow = StateGraph(RagState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()


def _langchain_messages(messages: list[dict[str, str]]) -> list[BaseMessage]:
    converted: list[BaseMessage] = []
    for message in messages[-20:]:
        content = str(message.get("content", ""))
        if message.get("role") == "assistant":
            converted.append(AIMessage(content=content))
        elif message.get("role") == "user":
            converted.append(HumanMessage(content=content))
    return converted


def _chunk_text(message_chunk: BaseMessage) -> str:
    content = getattr(message_chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(block.get("text", "")) if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content or "")


def verify_access_code(access_code: str, attempts: int):
    expected = os.getenv("APP_ACCESS_CODE", "")
    attempts = int(attempts or 0)
    if not expected:
        return False, attempts, gr.update(visible=True), gr.update(visible=False), gr.update(), _status(
            "管理员尚未配置 APP_ACCESS_CODE。", "error"
        )
    if attempts >= 5:
        return False, attempts, gr.update(visible=True), gr.update(visible=False), gr.update(), _status(
            "尝试次数过多，请刷新页面后重试。", "error"
        )
    if not hmac.compare_digest((access_code or "").encode(), expected.encode()):
        attempts += 1
        return False, attempts, gr.update(visible=True), gr.update(visible=False), gr.update(), _status(
            f"访问码错误，还可尝试 {max(0, 5 - attempts)} 次。", "error"
        )
    return True, 0, gr.update(visible=False), gr.update(visible=True), gr.update(value=""), _status(visible=False)


def connect_qwen(api_key: str, access_granted: bool):
    clean_key = (api_key or "").strip()
    if not access_granted:
        return _connect_failure("访问状态已失效，请刷新页面。", show_access=True)
    if len(clean_key) < 10:
        return _connect_failure("请输入有效的 Qwen API Key。")
    try:
        _chat_model(clean_key, validate_only=True).invoke([HumanMessage(content="只回复 OK")])
    except Exception as exc:
        return _connect_failure(_friendly_provider_error(exc, "Qwen Key 验证"))

    conversations, conversation_id = _initial_conversations()
    return (
        clean_key, [], conversations, conversation_id, DEFAULT_TITLE,
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True),
        gr.update(value=""), _status(visible=False), gr.update(value="● Qwen 已连接"),
        [], gr.update(value=DEFAULT_TITLE), _conversation_update(conversations, conversation_id),
        _document_update([]), _status(visible=False), gr.update(value="", interactive=True),
    )


def _connect_failure(message: str, *, show_access: bool = False):
    return (
        "", [], {}, None, DEFAULT_TITLE,
        gr.update(visible=show_access), gr.update(visible=not show_access), gr.update(visible=False),
        gr.update(visible=False), gr.update(), _status(message, "error"), gr.update(value=""),
        [], gr.update(value=DEFAULT_TITLE), gr.update(choices=[], value=None),
        gr.update(choices=[], value=None), _status(visible=False), gr.update(value="", interactive=True),
    )


def create_chat(conversations: dict[str, Any]):
    conversations = dict(conversations or {})
    if len(conversations) >= MAX_CONVERSATIONS:
        oldest = min(conversations, key=lambda key: conversations[key].get("created_at", 0))
        conversations.pop(oldest, None)
    conversation_id, conversation = _new_conversation()
    conversations[conversation_id] = conversation
    return (
        conversations, conversation_id, DEFAULT_TITLE, [], gr.update(value=DEFAULT_TITLE),
        _conversation_update(conversations, conversation_id), _status(visible=False),
        gr.update(value="", interactive=True),
    )


def select_chat(conversations: dict[str, Any], conversation_id: str):
    conversation = (conversations or {}).get(conversation_id)
    if not conversation:
        return None, DEFAULT_TITLE, [], gr.update(value=DEFAULT_TITLE), _status(
            "会话不存在，请新建会话。", "error"
        ), gr.update(value="", interactive=True)
    title = conversation.get("title") or DEFAULT_TITLE
    return conversation_id, title, list(conversation.get("messages", [])), gr.update(value=title), _status(
        visible=False
    ), gr.update(value="", interactive=True)


def delete_chat(conversations: dict[str, Any], conversation_id: str | None):
    conversations = dict(conversations or {})
    conversations.pop(conversation_id, None)
    if not conversations:
        next_id, next_conversation = _new_conversation()
        conversations[next_id] = next_conversation
    else:
        next_id = max(conversations, key=lambda key: conversations[key].get("created_at", 0))
    conversation = conversations[next_id]
    title = conversation.get("title") or DEFAULT_TITLE
    return (
        conversations, next_id, title, list(conversation.get("messages", [])),
        gr.update(value=title), _conversation_update(conversations, next_id),
        _status("会话已删除。", "success"), gr.update(value="", interactive=True),
    )


def upload_documents(file_values: Any, api_key: str, documents: list[dict[str, Any]]):
    documents = list(documents or [])
    files = file_values if isinstance(file_values, list) else ([file_values] if file_values else [])
    if not api_key:
        yield documents, _document_update(documents), gr.update(), _status("请先连接 Qwen。", "error")
        return
    if not files:
        yield documents, _document_update(documents), gr.update(), _status("请选择 PDF 或 TXT。", "error")
        return

    messages: list[str] = []
    existing_hashes = {document.get("sha256") for document in documents}
    for file_value in files:
        path, filename = _uploaded_path(file_value)
        if len(documents) >= MAX_DOCUMENTS:
            messages.append(f"{filename}：已达到每个会话 {MAX_DOCUMENTS} 个文档的上限")
            _remove_gradio_temp_file(path)
            continue
        yield documents, _document_update(documents), gr.update(), _status(f"正在解析并向量化 {filename}…")
        try:
            fingerprint = _sha256(path) if path.is_file() else ""
            if fingerprint and fingerprint in existing_hashes:
                raise DocumentError("该文件已经上传过。")
            document = _prepare_document(file_value, api_key)
            documents.append(document)
            existing_hashes.add(document["sha256"])
            messages.append(f"{filename}：上传成功")
        except DocumentError as exc:
            messages.append(f"{filename}：{exc}")
        except Exception as exc:
            logger.warning("Unexpected upload failure for %s: %s", filename, type(exc).__name__)
            messages.append(f"{filename}：处理失败，请稍后重试")
        finally:
            _remove_gradio_temp_file(path)
        yield documents, _document_update(documents), gr.update(), _status("；".join(messages))

    success_count = sum(message.endswith("上传成功") for message in messages)
    kind = "success" if success_count and success_count == len(messages) else ("info" if success_count else "error")
    yield documents, _document_update(documents), gr.update(value=None), _status("；".join(messages), kind)


def delete_document(document_id: str | None, documents: list[dict[str, Any]]):
    documents = list(documents or [])
    if not document_id:
        return documents, _document_update(documents), _status("请先选择要删除的文档。", "error")
    kept = [document for document in documents if document.get("id") != document_id]
    if len(kept) == len(documents):
        return documents, _document_update(documents), _status("文档不存在。", "error")
    return kept, _document_update(kept), _status("文档及其会话内向量已删除。", "success")


def send_message(
    user_message: str,
    conversations: dict[str, Any],
    conversation_id: str,
    api_key: str,
    documents: list[dict[str, Any]],
):
    clean_message = (user_message or "").strip()
    conversations = dict(conversations or {})
    conversation = conversations.get(conversation_id)
    current_history = list(conversation.get("messages", [])) if conversation else []
    title = conversation.get("title", DEFAULT_TITLE) if conversation else DEFAULT_TITLE

    if not clean_message:
        yield current_history, conversations, title, gr.update(value=title), _conversation_update(conversations, conversation_id), gr.update(
            value=user_message or "", interactive=True
        ), gr.update(interactive=True), _status("请输入消息后再发送。", "error")
        return
    if not api_key:
        yield current_history, conversations, title, gr.update(value=title), _conversation_update(conversations, conversation_id), gr.update(
            value=clean_message, interactive=True
        ), gr.update(interactive=True), _status("Qwen 连接已失效，请刷新页面重新连接。", "error")
        return
    if conversation is None:
        yield [], conversations, DEFAULT_TITLE, gr.update(value=DEFAULT_TITLE), _conversation_update(conversations, None), gr.update(
            value=clean_message, interactive=True
        ), gr.update(interactive=True), _status("当前会话不存在，请新建会话。", "error")
        return

    if title == DEFAULT_TITLE:
        title = clean_message[:24] + ("…" if len(clean_message) > 24 else "")
        conversation["title"] = title
    user_entry = {"role": "user", "content": clean_message}
    working_history = current_history + [user_entry]
    conversation["messages"] = working_history
    conversations[conversation_id] = conversation
    lookup_status = "正在检索当前文档…" if documents else "正在准备回答…"
    yield working_history, conversations, title, gr.update(value=title), _conversation_update(conversations, conversation_id), gr.update(
        value="", interactive=False
    ), gr.update(interactive=False), _status(lookup_status)

    raw_response = ""
    try:
        graph = _build_rag_graph(api_key, documents or [])
        for message_chunk, metadata in graph.stream(
            {"messages": _langchain_messages(working_history), "context": ""}, stream_mode="messages"
        ):
            if (metadata or {}).get("langgraph_node") != "generate":
                continue
            piece = _chunk_text(message_chunk)
            if not piece:
                continue
            raw_response += piece
            streamed_history = working_history + [
                {"role": "assistant", "content": _format_response(raw_response)}
            ]
            conversation["messages"] = streamed_history
            conversations[conversation_id] = conversation
            yield streamed_history, conversations, title, gr.update(value=title), _conversation_update(
                conversations, conversation_id
            ), gr.update(value="", interactive=False), gr.update(interactive=False), _status(visible=False)
    except Exception as exc:
        error_message = _friendly_provider_error(exc, "回答生成")
        raw_response = (
            f"{raw_response}\n\n> ⚠️ {error_message}" if raw_response else error_message
        )
        final_status = _status(error_message, "error")
    else:
        if not raw_response:
            raw_response = "Qwen 没有返回有效内容，请稍后重试。"
            final_status = _status("没有收到有效回复。", "error")
        else:
            final_status = _status(visible=False)

    final_history = working_history + [{"role": "assistant", "content": _format_response(raw_response)}]
    conversation["messages"] = final_history
    conversations[conversation_id] = conversation
    yield final_history, conversations, title, gr.update(value=title), _conversation_update(conversations, conversation_id), gr.update(
        value="", interactive=True
    ), gr.update(interactive=True), final_status


APP_CSS = r"""
:root{--bg:#212121;--side:#171717;--surface:#2f2f2f;--border:#454545;--text:#ececec;--muted:#a8a8a8;--accent:#10a37f;--accent2:#0d8f70}
html,body{background:var(--bg)!important;overflow:hidden}.gradio-container{width:100%!important;max-width:none!important;min-height:100dvh!important;margin:0!important;padding:0!important;background:var(--bg)!important;color:var(--text)!important;font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif!important}.gradio-container main.fillable{height:100dvh!important;min-height:100dvh!important;padding:0!important}footer,.footer{display:none!important}
#access-page,#key-page{min-height:100dvh!important;justify-content:center!important;align-items:center!important;padding:24px!important;background:radial-gradient(circle at 50% 18%,rgba(16,163,127,.11),transparent 32%),var(--bg)!important}.setup-card{width:min(440px,100%)!important;gap:17px!important;padding:34px!important;border:1px solid var(--border)!important;border-radius:20px!important;background:#282828!important;box-shadow:0 24px 60px rgba(0,0,0,.28)!important}.brand-lockup{text-align:center}.brand-mark{display:inline-grid;place-items:center;width:48px;height:48px;margin-bottom:14px;border-radius:14px;background:var(--accent);color:#fff;font-size:22px;font-weight:700}.brand-lockup h1{margin:0;color:var(--text);font-size:25px}.brand-lockup p{margin:8px 0 0;color:var(--muted);font-size:14px;line-height:1.6}.setup-heading{text-align:center}.setup-heading h2{margin:0!important;color:var(--text);font-size:18px}.setup-card [data-testid="block-info"]{padding:0 0 7px!important;background:transparent!important;color:#cfcfcf!important;font-size:13px!important}.setup-card input{min-height:48px!important;border:1px solid var(--border)!important;border-radius:12px!important;background:#1f1f1f!important;color:var(--text)!important}.setup-card input:focus{border-color:var(--accent)!important;box-shadow:0 0 0 3px rgba(16,163,127,.18)!important}.setup-note{color:var(--muted)!important;font-size:12px!important;line-height:1.55!important}.setup-note p{margin:0!important}.primary-action,.primary-action button{min-height:46px!important;border:0!important;border-radius:12px!important;background:var(--accent)!important;color:#fff!important;font-weight:650!important}.primary-action:hover,.primary-action button:hover{background:var(--accent2)!important}.text-action,.text-action button{border:0!important;background:transparent!important;color:#7ed9c2!important;box-shadow:none!important;font-size:14px!important}.inline-status{min-height:22px!important;color:var(--muted)!important;font-size:13px!important;text-align:center}
#app-sidebar{height:100dvh!important;padding:12px 10px!important;border-right:1px solid #252525!important;background:var(--side)!important}.sidebar-brand{display:flex;align-items:center;gap:10px;padding:8px 10px 12px;color:var(--text);font-weight:650}.sidebar-brand-mark{display:inline-grid;place-items:center;width:30px;height:30px;border-radius:9px;background:var(--accent);color:#fff}#new-chat,#new-chat button{justify-content:flex-start!important;min-height:42px!important;padding:0 12px!important;border:1px solid #3a3a3a!important;border-radius:10px!important;background:transparent!important;color:var(--text)!important;box-shadow:none!important}#new-chat:hover,#new-chat button:hover{background:#252525!important}.sidebar-section-title{margin:14px 10px 5px!important;color:#8b8b8b!important;font-size:12px!important;font-weight:600!important}#conversation-list,#document-list{flex:0 1 auto!important;max-height:25dvh!important;overflow-y:auto!important;border:0!important;background:transparent!important;scrollbar-width:thin;scrollbar-color:#555 transparent}#conversation-list .wrap,#document-list .wrap{gap:3px!important}#conversation-list label,#document-list label{min-height:36px!important;padding:8px 10px!important;border:0!important;border-radius:9px!important;background:transparent!important;color:#cfcfcf!important;font-size:13px!important;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}#conversation-list label:hover,#document-list label:hover{background:#262626!important}#conversation-list label:has(input:checked),#document-list label:has(input:checked){background:#303030!important;color:#fff!important}#conversation-list input,#document-list input{display:none!important}.sidebar-actions{gap:6px!important}.sidebar-small,.sidebar-small button{min-height:34px!important;border:1px solid #393939!important;border-radius:9px!important;background:#242424!important;color:#cfcfcf!important;box-shadow:none!important;font-size:12px!important}.sidebar-small:hover,.sidebar-small button:hover{background:#303030!important}.danger-action,.danger-action button{color:#d8a5a5!important}#upload-files{max-height:112px!important;border:1px dashed #444!important;border-radius:10px!important;background:#202020!important}#upload-files label{font-size:12px!important}.sidebar-status{min-height:18px!important;color:#999!important;font-size:11px!important}.sidebar-status p{margin:0!important}#sidebar-bottom{margin-top:auto!important;gap:4px!important;padding-top:9px!important;border-top:1px solid #2c2c2c!important}#key-badge{padding:5px 8px!important;color:#78d4bb!important;font-size:12px!important}
#chat-page{height:100dvh!important;min-height:100dvh!important;gap:0!important;background:var(--bg)!important}#chat-header{flex:0 0 56px!important;align-items:center!important;padding:0 22px!important;border-bottom:1px solid rgba(255,255,255,.05)!important;background:rgba(33,33,33,.94)!important}.sidebar-parent:not(:has(#app-sidebar.open)) #chat-header{padding-left:58px!important}#title-display{min-width:0!important;color:var(--text)!important;font-size:15px!important;font-weight:600!important}#title-display p{margin:0!important;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}#chat-content{width:100%!important;max-width:920px!important;flex:1 1 auto!important;min-height:0!important;gap:8px!important;margin:0 auto!important;padding:0 28px 14px!important}#chatbot{flex:1 1 auto!important;min-height:0!important;border:0!important;background:transparent!important;box-shadow:none!important}#chatbot .wrap,#chatbot .bubble-wrap{background:transparent!important}#chatbot .bubble-wrap{padding-right:14px!important;scrollbar-gutter:stable;scrollbar-width:thin;scrollbar-color:#555 transparent}#chatbot .bubble-wrap::-webkit-scrollbar{width:8px}#chatbot .bubble-wrap::-webkit-scrollbar-track{background:transparent}#chatbot .bubble-wrap::-webkit-scrollbar-thumb{border:2px solid transparent;border-radius:999px;background:#555;background-clip:padding-box}#chatbot .bubble-wrap::-webkit-scrollbar-thumb:hover{background:#707070;background-clip:padding-box}#chatbot .icon-button-wrapper.top-panel{top:10px!important;right:28px!important;border-radius:8px!important}#chatbot .message{border:0!important;box-shadow:none!important;line-height:1.7!important}#chatbot .message-row.user-row.bubble{width:fit-content!important;max-width:min(80%,680px)!important;margin-left:auto!important}#chatbot .message-row.user-row.bubble .message.user{width:auto!important;max-width:none!important;padding:10px 15px!important;border-radius:18px!important;background:var(--surface)!important;color:var(--text)!important}#chatbot .message.user .prose{overflow-wrap:anywhere!important;word-break:normal!important}#chatbot .message.bot{width:100%!important;max-width:100%!important;padding:10px 2px!important;background:transparent!important;color:var(--text)!important}#response-status{min-height:20px!important;overflow:hidden!important;color:var(--muted)!important;font-size:12px!important;text-align:center}#composer-shell{flex:0 0 auto!important;align-items:center!important;gap:8px!important;padding:8px 10px 8px 16px!important;border:1px solid #4a4a4a!important;border-radius:24px!important;background:var(--surface)!important;box-shadow:0 8px 28px rgba(0,0,0,.18)!important}#composer-shell:focus-within{border-color:#626262!important}#message-box,#message-box .form,#message-box .wrap{border:0!important;background:transparent!important;box-shadow:none!important}#message-box .input-container{display:flex!important;align-items:center!important;min-height:42px!important}#message-box textarea{min-height:42px!important;max-height:150px!important;padding:10px 0!important;border:0!important;background:transparent!important;color:var(--text)!important;box-shadow:none!important;line-height:22px!important}#message-box textarea::placeholder{color:#949494!important}#send-button{flex:0 0 42px!important;min-width:42px!important;width:42px!important;height:42px!important;padding:0!important;border:0!important;border-radius:50%!important;background:var(--accent)!important;color:#fff!important;box-shadow:none!important;font-size:17px!important}#send-button:hover{background:var(--accent2)!important}#send-button:disabled{background:#565656!important;color:#a8a8a8!important}#chat-hint{overflow:hidden!important;color:#858585!important;font-size:11px!important;text-align:center}#chat-hint p{margin:0!important}@media(max-width:700px){#access-page,#key-page{padding:14px!important}.setup-card{padding:26px 20px!important;border-radius:16px!important}#chat-header{padding:0 14px 0 52px!important}#chat-content{padding:0 12px 10px!important}#chatbot .message-row.user-row.bubble{max-width:88%!important}#composer-shell{border-radius:20px!important}}
"""

APP_THEME = gr.themes.Soft(
    primary_hue="emerald", secondary_hue="emerald", neutral_hue="gray",
    font=["ui-sans-serif", "system-ui", "sans-serif"],
)


with gr.Blocks(title="智能文档助手", theme=APP_THEME, css=APP_CSS, fill_height=True, fill_width=True) as demo:
    access_granted_state = gr.State(False, time_to_live=SESSION_TTL_SECONDS)
    access_attempts_state = gr.State(0, time_to_live=SESSION_TTL_SECONDS)
    api_key_state = gr.State("", time_to_live=SESSION_TTL_SECONDS, delete_callback=_release_state)
    documents_state = gr.State([], time_to_live=SESSION_TTL_SECONDS, delete_callback=_release_state)
    conversations_state = gr.State({}, time_to_live=SESSION_TTL_SECONDS, delete_callback=_release_state)
    current_conversation_state = gr.State(None, time_to_live=SESSION_TTL_SECONDS)
    conversation_title_state = gr.State(DEFAULT_TITLE, time_to_live=SESSION_TTL_SECONDS)

    with gr.Sidebar(open=True, visible=False, width=290, elem_id="app-sidebar") as sidebar:
        gr.HTML("<div class='sidebar-brand'><span class='sidebar-brand-mark'>✦</span>智能文档助手</div>", padding=False)
        new_chat_button = gr.Button("＋  新建聊天", size="sm", elem_id="new-chat")
        gr.Markdown("最近对话", elem_classes="sidebar-section-title")
        conversation_list = gr.Radio(choices=[], value=None, show_label=False, container=False, elem_id="conversation-list")
        delete_chat_button = gr.Button("删除当前会话", size="sm", elem_classes=["sidebar-small", "danger-action"])
        gr.Markdown("会话文档（最多 3 个）", elem_classes="sidebar-section-title")
        document_list = gr.Radio(choices=[], value=None, show_label=False, container=False, elem_id="document-list")
        upload_files = gr.File(label="拖入 PDF / TXT", file_count="multiple", file_types=[".pdf", ".txt"], type="filepath", elem_id="upload-files")
        with gr.Row(elem_classes="sidebar-actions"):
            upload_button = gr.Button("上传并解析", size="sm", elem_classes="sidebar-small")
            delete_document_button = gr.Button("删除文档", size="sm", elem_classes=["sidebar-small", "danger-action"])
        document_status = gr.Markdown("", visible=False, elem_classes="sidebar-status")
        with gr.Column(elem_id="sidebar-bottom"):
            key_badge = gr.Markdown("", elem_id="key-badge")
            clear_session_button = gr.Button("清除本次会话", size="sm", elem_classes=["sidebar-small", "danger-action"])

    with gr.Column(visible=True, elem_id="access-page") as access_page:
        with gr.Column(elem_classes="setup-card"):
            gr.HTML("<div class='brand-lockup'><div class='brand-mark'>✦</div><h1>智能文档助手</h1><p>输入演示访问码后继续</p></div>", padding=False)
            gr.Markdown("## 访问验证", elem_classes="setup-heading")
            access_code = gr.Textbox(label="访问码", placeholder="请输入访问码", type="password", autofocus=True)
            access_status = gr.Markdown("", visible=False, elem_classes="inline-status")
            access_button = gr.Button("继续", variant="primary", elem_classes="primary-action")
            gr.Markdown("访问码仅用于限制演示入口，不是账号系统。刷新页面后需要重新输入。", elem_classes="setup-note")

    with gr.Column(visible=False, elem_id="key-page") as key_page:
        with gr.Column(elem_classes="setup-card"):
            gr.HTML("<div class='brand-lockup'><div class='brand-mark'>✦</div><h1>连接 Qwen</h1><p>使用你自己的阿里云百炼 API Key</p></div>", padding=False)
            gr.Markdown("## Qwen API Key", elem_classes="setup-heading")
            key_input = gr.Textbox(label="API Key", placeholder="sk-...", type="password")
            key_status = gr.Markdown("", visible=False, elem_classes="inline-status")
            connect_button = gr.Button("验证并进入", variant="primary", elem_classes="primary-action")
            back_button = gr.Button("返回访问验证", size="sm", elem_classes="text-action")
            gr.Markdown("Key 只保存在本次服务端会话内，不写入数据库、文件或日志。建议使用单独创建并限制额度的 Key。", elem_classes="setup-note")

    with gr.Column(visible=False, elem_id="chat-page") as chat_page:
        with gr.Row(elem_id="chat-header"):
            title_display = gr.Markdown(DEFAULT_TITLE, elem_id="title-display")
        with gr.Column(elem_id="chat-content"):
            chatbot = gr.Chatbot(
                type="messages", show_label=False, height="100%", min_height=320, layout="bubble",
                show_copy_button=True, render_markdown=True,
                placeholder="### 今天想了解什么？\n\n可以直接提问，或先在左侧上传 PDF / TXT。", elem_id="chatbot",
            )
            response_status = gr.Markdown("", visible=False, elem_id="response-status")
            with gr.Row(elem_id="composer-shell"):
                message = gr.Textbox(
                    label="消息", show_label=False, placeholder="给智能文档助手发送消息",
                    lines=1, max_lines=6, container=False, scale=1, elem_id="message-box",
                )
                send_button = gr.Button("➤", variant="primary", size="sm", min_width=42, elem_id="send-button")
            gr.Markdown("内容由 AI 生成，请结合原文和实际情况判断。", elem_id="chat-hint")

    reset_outputs = [
        access_granted_state, access_attempts_state, api_key_state, documents_state,
        conversations_state, current_conversation_state, conversation_title_state,
        access_page, key_page, chat_page, sidebar, access_code, access_status, key_input,
        key_status, key_badge, chatbot, title_display, conversation_list, document_list,
        upload_files, document_status, response_status, message, send_button,
    ]

    def reset_session():
        return (
            False, 0, "", [], {}, None, DEFAULT_TITLE,
            gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
            gr.update(value=""), _status(visible=False), gr.update(value=""), _status(visible=False),
            gr.update(value=""), [], gr.update(value=DEFAULT_TITLE), gr.update(choices=[], value=None),
            gr.update(choices=[], value=None), gr.update(value=None), _status(visible=False), _status(visible=False),
            gr.update(value="", interactive=True), gr.update(interactive=True),
        )

    access_outputs = [access_granted_state, access_attempts_state, access_page, key_page, access_code, access_status]
    access_button.click(verify_access_code, [access_code, access_attempts_state], access_outputs, show_progress="hidden", concurrency_limit=1)
    access_code.submit(verify_access_code, [access_code, access_attempts_state], access_outputs, show_progress="hidden", concurrency_limit=1)

    connect_outputs = [
        api_key_state, documents_state, conversations_state, current_conversation_state,
        conversation_title_state, access_page, key_page, chat_page, sidebar, key_input,
        key_status, key_badge, chatbot, title_display, conversation_list, document_list,
        response_status, message,
    ]
    connect_button.click(connect_qwen, [key_input, access_granted_state], connect_outputs, show_progress="minimal", concurrency_limit=1)
    key_input.submit(connect_qwen, [key_input, access_granted_state], connect_outputs, show_progress="minimal", concurrency_limit=1)

    back_button.click(reset_session, outputs=reset_outputs, show_progress="hidden")
    clear_session_button.click(reset_session, outputs=reset_outputs, show_progress="hidden")
    demo.load(reset_session, outputs=reset_outputs, show_progress="hidden", queue=False)

    chat_change_outputs = [
        conversations_state, current_conversation_state, conversation_title_state, chatbot,
        title_display, conversation_list, response_status, message,
    ]
    new_chat_button.click(create_chat, conversations_state, chat_change_outputs, show_progress="hidden", concurrency_limit=1)
    delete_chat_button.click(delete_chat, [conversations_state, current_conversation_state], chat_change_outputs, show_progress="hidden", concurrency_limit=1)
    conversation_list.change(
        select_chat, [conversations_state, conversation_list],
        [current_conversation_state, conversation_title_state, chatbot, title_display, response_status, message],
        show_progress="hidden", concurrency_limit=1,
    )

    upload_button.click(
        upload_documents, [upload_files, api_key_state, documents_state],
        [documents_state, document_list, upload_files, document_status], show_progress="minimal",
        concurrency_limit=1, concurrency_id="document-change",
    )
    delete_document_button.click(
        delete_document, [document_list, documents_state],
        [documents_state, document_list, document_status], show_progress="hidden",
        concurrency_limit=1, concurrency_id="document-change",
    )

    chat_inputs = [message, conversations_state, current_conversation_state, api_key_state, documents_state]
    chat_outputs = [
        chatbot, conversations_state, conversation_title_state, title_display,
        conversation_list, message, send_button, response_status,
    ]
    send_button.click(
        send_message, chat_inputs, chat_outputs, trigger_mode="once", concurrency_limit=1,
        concurrency_id="chat-submit", show_progress="hidden", stream_every=0.08,
    )
    message.submit(
        send_message, chat_inputs, chat_outputs, trigger_mode="once", concurrency_limit=1,
        concurrency_id="chat-submit", show_progress="hidden", stream_every=0.08,
    )


if __name__ == "__main__":
    demo.queue(max_size=20, default_concurrency_limit=2).launch(
        server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")), show_api=False,
        max_file_size="10mb", state_session_capacity=20, enable_monitoring=False,
    )
