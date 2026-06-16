import base64
import hashlib
import asyncio
import contextvars
import html
import json
import os
import re
import secrets as token_secrets
import shutil
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from queue import Empty, Queue
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote_plus, unquote, urlencode, urlparse
from urllib.request import Request, urlopen

import streamlit as st
import streamlit.components.v1 as components
from agents import (
    Agent,
    ModelSettings,
    OpenAIChatCompletionsModel,
    Runner,
    SQLiteSession,
    function_tool,
    set_tracing_disabled,
)
from openai import AsyncOpenAI
from openai.types.shared import Reasoning


APP_TITLE = "Life Coach Agent"
APP_SHORT_TITLE = "Life Coach"
APP_ICON_PATH = Path(__file__).with_name("static") / "icons" / "icon-192.png"
DEFAULT_MODEL = "deepseek-v4-flash"
SUPPORTED_MODELS = (DEFAULT_MODEL, "deepseek-v4-pro")
MODEL_LABELS = {
    "deepseek-v4-flash": "Flash",
    "deepseek-v4-pro": "Pro",
}
DEFAULT_THINKING_MODE = "fast"
DEFAULT_COACHING_STYLE = "balanced"
CUSTOM_INSTRUCTIONS_MAX_CHARS = 1200
SESSION_QUERY_PARAM = "session"
SHARE_QUERY_PARAM = "share"
AUTH_CALLBACK_QUERY_PARAM = "auth"
OAUTH_STATE_TTL_MINUTES = 10
OAUTH_URL_CACHE_VERSION = "no-custom-state-v2"
AUTH_COOKIE_NAME = "life_coach_auth"
AUTH_SESSION_DAYS = 30
MAX_SEARCH_CALLS_PER_MESSAGE = 2
THINKING_MODES: dict[str, dict[str, str | None]] = {
    "fast": {
        "label": "빠른 응답",
        "description": "thinking off",
        "thinking_type": "disabled",
        "effort": None,
    },
    "high": {
        "label": "깊은 생각",
        "description": "thinking high",
        "thinking_type": "enabled",
        "effort": "high",
    },
    "xhigh": {
        "label": "최대 생각",
        "description": "thinking max",
        "thinking_type": "enabled",
        "effort": "xhigh",
    },
}
COACHING_STYLES: dict[str, dict[str, str]] = {
    "balanced": {
        "label": "균형",
        "description": "따뜻하지만 실행 중심으로 답합니다.",
        "instructions": (
            "Use a balanced coaching tone: warm, practical, and concise. "
            "Start with brief empathy, then give concrete next steps."
        ),
    },
    "gentle": {
        "label": "다정",
        "description": "부담을 낮추고 부드럽게 격려합니다.",
        "instructions": (
            "Use a gentle and reassuring tone. Reduce pressure, validate the "
            "user's feelings, and suggest very small first steps."
        ),
    },
    "direct": {
        "label": "직설",
        "description": "돌려 말하지 않고 핵심과 행동을 짚습니다.",
        "instructions": (
            "Use a direct and candid tone without being harsh. Avoid long "
            "comforting preambles and focus on the highest-leverage actions."
        ),
    },
    "accountability": {
        "label": "실행관리",
        "description": "체크리스트와 다음 행동을 강하게 잡아줍니다.",
        "instructions": (
            "Act like an accountability coach. Convert advice into a short "
            "checklist, ask for a commitment, and propose a follow-up action."
        ),
    },
    "analytical": {
        "label": "분석",
        "description": "원인 분석과 실험 설계를 더 강조합니다.",
        "instructions": (
            "Use an analytical coaching style. Identify likely causes, separate "
            "assumptions from facts, and suggest a small experiment."
        ),
    },
}
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DB_PATH = Path(__file__).with_name("life_coach_sessions.db")
GOALS_PATH = Path(__file__).with_name("goals") / "personal_goals.md"
GOALS_MAX_CHARS = 20000
GOALS_PREVIEW_CHARS = 600
MAX_GOAL_SEARCH_CALLS_PER_MESSAGE = 2
MOVIE_AGENT_ENV_PATH = Path.home() / "Documents" / "movie-agent" / ".env"
KST = timezone(timedelta(hours=9), "KST")
SEARCH_TIMINGS: contextvars.ContextVar[list[dict[str, object]] | None] = (
    contextvars.ContextVar("SEARCH_TIMINGS", default=None)
)
RUN_EVENTS: contextvars.ContextVar[list[dict[str, object]] | None] = (
    contextvars.ContextVar("RUN_EVENTS", default=None)
)
RUN_STARTED_AT: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "RUN_STARTED_AT",
    default=None,
)
RUN_EVENT_RENDERER: contextvars.ContextVar[
    Callable[[list[dict[str, object]]], None] | None
] = contextvars.ContextVar("RUN_EVENT_RENDERER", default=None)
RUN_EVENT_QUEUE: contextvars.ContextVar[Queue | None] = contextvars.ContextVar(
    "RUN_EVENT_QUEUE",
    default=None,
)
SEARCH_CALL_COUNT: contextvars.ContextVar[list[int] | None] = contextvars.ContextVar(
    "SEARCH_CALL_COUNT",
    default=None,
)
GOALS_TIMINGS: contextvars.ContextVar[list[dict[str, object]] | None] = (
    contextvars.ContextVar("GOALS_TIMINGS", default=None)
)
GOAL_SEARCH_CALL_COUNT: contextvars.ContextVar[list[int] | None] = (
    contextvars.ContextVar("GOAL_SEARCH_CALL_COUNT", default=None)
)
STOP_EVENTS: dict[str, threading.Event] = {}
WEB_SEARCH_HINTS = (
    "아침",
    "일찍",
    "알람",
    "스누즈",
    "습관",
    "루틴",
    "동기",
    "자기계발",
    "집중",
    "공부",
    "수면",
    "생산성",
    "목표",
    "운동",
    "일기",
    "진행",
    "팁",
    "방법",
    "조언",
    "검색",
    "찾아",
    "habit",
    "routine",
    "motivation",
    "focus",
    "productivity",
    "sleep",
    "goal",
    "tips",
    "search",
)

set_tracing_disabled(True)
os.environ.setdefault("OPENAI_AGENTS_TRACE_INCLUDE_SENSITIVE_DATA", "0")


class GenerationStopped(Exception):
    """Raised when the user asks to stop the current response."""


def request_stop(run_id: str) -> None:
    stop_event = STOP_EVENTS.get(run_id)
    if stop_event:
        stop_event.set()
    st.session_state.stop_requested = True


def ensure_not_stopped(stop_event: threading.Event | None) -> None:
    if stop_event and stop_event.is_set():
        raise GenerationStopped


LIFE_COACH_INSTRUCTIONS = """
You are a warm, practical life coach for Korean users.

Your job:
- Encourage the user without exaggerating or sounding generic.
- Give concrete advice about motivation, self-development, habits, routines,
  productivity, reflection, and goal setting.
- Use the search_web tool when the user asks for advice that can benefit from
  current or evidence-informed tips, especially about motivation content,
  self-development methods, habit formation, routines, sleep, focus, or learning.
- For concrete how-to questions such as waking up early, stopping snooze,
  building habits, staying motivated, focusing, or improving routines, call
  search_web before your final answer even if the user does not explicitly ask
  for a web search.
- Use search_web once when one focused query is enough. You may call it a
  second time only when a distinct angle would improve the answer, such as a
  Korean practical query plus an English evidence-oriented query. Do not repeat
  substantially similar searches.
- When you use search_web, synthesize the results in your own words and mention
  the most useful source names or URLs briefly.
- When mentioning web sources, format them as Markdown links, for example
  [source name](https://example.com). Do not leave source URLs as plain text.
- Keep the response in Korean unless the user asks for another language.
- Keep answers structured and actionable: empathize briefly, then give 3-5
  practical steps the user can try today.
- Do not present yourself as a therapist, doctor, or medical professional.
- For mental health crisis, self-harm, or medical issues, respond supportively
  and recommend professional/local emergency help instead of coaching only.
- Remember prior user goals and preferences through session memory.
"""

SEARCH_AGENT_INSTRUCTIONS = """
You are a research planner for a Korean life coach.

You may have up to two tools:
- search_goals: search the user's personal goal and journal document.
- search_web: search the public web for tips and evidence.

Your job:
- If the search_goals tool is available, call it FIRST to recall the user's
  goals, plans, and past progress that are relevant to the question.
- Then, when current or evidence-informed tips would help, call search_web
  once. Call search_web a second time only for a clearly different angle, such
  as practical Korean tips plus evidence-oriented English sources.
- Do not repeat substantially similar searches.
- After the tool results are returned, respond with only: SEARCH_DONE
"""

STREAMING_COACH_INSTRUCTIONS = """
You are a warm, practical life coach for Korean users.

The app may provide the user's personal goal/journal excerpts and web search
results inside the user message. When such context is provided, use it and do
not ask for another search.

Your job:
- Encourage the user without exaggerating or sounding generic.
- Give concrete advice about motivation, self-development, habits, routines,
  productivity, reflection, and goal setting.
- When personal goal or journal context is provided, reference it directly:
  compare the user's stated goals with their recent progress, acknowledge what
  is going well, point out where they are slipping, and tailor next steps to
  their situation. Track progress over time when journal dates are available.
- Keep the response in Korean unless the user asks for another language.
- Keep answers structured and actionable: empathize briefly, then give 3-5
  practical steps the user can try today.
- Mention useful source names or URLs briefly when search results are provided.
- Format every source as a Markdown link, for example
  [source name](https://example.com). Do not leave source URLs as plain text.
- Do not present yourself as a therapist, doctor, or medical professional.
- For mental health crisis, self-harm, or medical issues, respond supportively
  and recommend professional/local emergency help instead of coaching only.
- Remember prior user goals and preferences through session memory.
"""


class SearchResultParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self.current: dict[str, str] | None = None
        self.capturing_title = False
        self.capturing_snippet = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key: value or "" for key, value in attrs}
        class_name = attrs_dict.get("class", "")

        if tag == "a" and "result__a" in class_name:
            self._flush_current()
            self.current = {
                "title": "",
                "url": normalize_duckduckgo_url(attrs_dict.get("href", "")),
                "snippet": "",
            }
            self.capturing_title = True
            return

        if self.current and "result__snippet" in class_name:
            self.capturing_snippet = True

    def handle_data(self, data: str) -> None:
        if not self.current:
            return

        text = " ".join(data.split())
        if not text:
            return

        if self.capturing_title:
            self.current["title"] = (self.current["title"] + " " + text).strip()
        elif self.capturing_snippet:
            self.current["snippet"] = (self.current["snippet"] + " " + text).strip()

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self.capturing_title = False
        elif tag in {"a", "div"}:
            self.capturing_snippet = False

    def close(self) -> None:
        super().close()
        self._flush_current()

    def _flush_current(self) -> None:
        if self.current and self.current["title"] and self.current["url"]:
            self.results.append(self.current)
        self.current = None
        self.capturing_title = False
        self.capturing_snippet = False


def normalize_duckduckgo_url(raw_url: str) -> str:
    if raw_url.startswith("//"):
        raw_url = f"https:{raw_url}"

    parsed = urlparse(raw_url)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        redirect_url = parse_qs(parsed.query).get("uddg", [""])[0]
        if redirect_url:
            return unquote(redirect_url)

    return raw_url


def read_env_file_value(path: Path, key: str) -> str | None:
    if not path.exists():
        return None

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        name, value = stripped.split("=", 1)
        if name.strip() != key:
            continue

        value = value.strip().strip('"').strip("'")
        return value or None

    return None


def read_deepseek_api_key() -> str | None:
    """Read the API key without displaying or logging the secret value."""
    env_key = os.getenv("DEEPSEEK_API_KEY")
    if env_key:
        return env_key

    try:
        secret_key = st.secrets.get("DEEPSEEK_API_KEY")
    except Exception:
        secret_key = None

    if secret_key:
        return secret_key

    return read_env_file_value(MOVIE_AGENT_ENV_PATH, "DEEPSEEK_API_KEY")


def read_config_value(name: str) -> str | None:
    env_value = os.getenv(name)
    if env_value:
        return env_value

    try:
        secret_value = st.secrets.get(name)
    except Exception:
        secret_value = None

    if secret_value:
        return str(secret_value)

    return None


def read_supabase_config() -> dict[str, str] | None:
    url = read_config_value("SUPABASE_URL")
    key = read_config_value("SUPABASE_SERVICE_ROLE_KEY") or read_config_value(
        "SUPABASE_ANON_KEY"
    )
    if not url or not key:
        return None

    return {"url": url.rstrip("/"), "key": key}


def read_supabase_public_config() -> dict[str, str] | None:
    url = read_config_value("SUPABASE_URL")
    key = read_config_value("SUPABASE_ANON_KEY")
    if not url or not key:
        return None

    return {"url": url.rstrip("/"), "key": key}


def read_app_base_url() -> str:
    return (read_config_value("APP_BASE_URL") or "http://localhost:8501").rstrip("/")


def app_runs_on_https() -> bool:
    return read_app_base_url().startswith("https://")


def make_app_auth_token() -> str:
    return token_secrets.token_urlsafe(48)


def hash_app_auth_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def get_auth_cookie_token() -> str | None:
    try:
        cookies = st.context.cookies
    except Exception:
        return None

    token = cookies.get(AUTH_COOKIE_NAME) if cookies else None
    if not isinstance(token, str):
        return None
    if not re.fullmatch(r"[A-Za-z0-9_-]{40,160}", token):
        return None
    return token


def supabase_request(
    method: str,
    path: str,
    payload: object | None = None,
    prefer: str | None = None,
) -> object:
    config = read_supabase_config()
    if not config:
        raise RuntimeError("Supabase is not configured")

    data = None
    headers = {
        "Accept": "application/json",
        "apikey": config["key"],
        "Authorization": f"Bearer {config['key']}",
    }
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    if prefer:
        headers["Prefer"] = prefer

    request = Request(
        f"{config['url']}/rest/v1/{path.lstrip('/')}",
        data=data,
        headers=headers,
        method=method,
    )

    try:
        with urlopen(request, timeout=10) as response:
            response_text = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        raise RuntimeError(f"Supabase HTTP {exc.code}") from exc
    except (OSError, URLError) as exc:
        raise RuntimeError(f"Supabase request failed: {exc.__class__.__name__}") from exc

    if not response_text:
        return None
    return json.loads(response_text)


def supabase_auth_request(
    method: str,
    path: str,
    payload: object | None = None,
    access_token: str | None = None,
) -> object:
    config = read_supabase_public_config()
    if not config:
        raise RuntimeError("Supabase public config is not configured")

    data = None
    headers = {
        "Accept": "application/json",
        "apikey": config["key"],
        "Authorization": f"Bearer {access_token or config['key']}",
    }
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")

    request = Request(
        f"{config['url']}/auth/v1/{path.lstrip('/')}",
        data=data,
        headers=headers,
        method=method,
    )

    try:
        with urlopen(request, timeout=15) as response:
            response_text = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        raise RuntimeError(f"Supabase Auth HTTP {exc.code}") from exc
    except (OSError, URLError) as exc:
        raise RuntimeError(
            f"Supabase Auth request failed: {exc.__class__.__name__}"
        ) from exc

    if not response_text:
        return None
    return json.loads(response_text)


def supabase_refresh_auth_session(refresh_token: str) -> dict[str, object]:
    response = supabase_auth_request(
        "POST",
        "token?grant_type=refresh_token",
        {"refresh_token": refresh_token},
    )
    if not isinstance(response, dict) or not response.get("user"):
        raise RuntimeError("Supabase Auth refresh failed")
    return response


def supabase_revoke_refresh_token(refresh_token: str) -> None:
    try:
        supabase_auth_request("POST", "logout", {"refresh_token": refresh_token})
    except Exception:
        return


def supabase_create_app_auth_session(user_id: str, refresh_token: str) -> str:
    token = make_app_auth_token()
    expires_at = datetime.now(timezone.utc) + timedelta(days=AUTH_SESSION_DAYS)
    supabase_request(
        "POST",
        "life_coach_auth_sessions",
        {
            "token_hash": hash_app_auth_token(token),
            "user_id": user_id,
            "refresh_token": refresh_token,
            "expires_at": expires_at.isoformat(),
        },
        prefer="return=minimal",
    )
    return token


def supabase_load_app_auth_session(token: str) -> dict[str, object] | None:
    params = urlencode(
        {
            "select": "token_hash,user_id,refresh_token,expires_at,revoked_at",
            "token_hash": f"eq.{hash_app_auth_token(token)}",
            "expires_at": f"gt.{datetime.now(timezone.utc).isoformat()}",
            "revoked_at": "is.null",
            "limit": "1",
        }
    )
    response = supabase_request("GET", f"life_coach_auth_sessions?{params}")
    if isinstance(response, list) and response:
        return response[0]
    return None


def supabase_update_app_auth_session(
    token: str,
    user_id: str,
    refresh_token: str,
) -> None:
    expires_at = datetime.now(timezone.utc) + timedelta(days=AUTH_SESSION_DAYS)
    supabase_request(
        "PATCH",
        f"life_coach_auth_sessions?token_hash=eq.{hash_app_auth_token(token)}",
        {
            "user_id": user_id,
            "refresh_token": refresh_token,
            "expires_at": expires_at.isoformat(),
            "revoked_at": None,
        },
        prefer="return=minimal",
    )


def supabase_revoke_app_auth_session(token: str | None) -> None:
    if not token:
        return
    try:
        supabase_request(
            "PATCH",
            f"life_coach_auth_sessions?token_hash=eq.{hash_app_auth_token(token)}",
            {"revoked_at": datetime.now(timezone.utc).isoformat()},
            prefer="return=minimal",
        )
    except Exception:
        return


def make_chat_session_key() -> str:
    return f"life-coach-{uuid.uuid4().hex}"


def get_query_session_key() -> str | None:
    try:
        value = st.query_params.get(SESSION_QUERY_PARAM)
    except Exception:
        return None

    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, str) and re.fullmatch(r"life-coach-[a-f0-9]{32}", value):
        return value
    return None


def get_query_share_token() -> str | None:
    value = get_query_value(SHARE_QUERY_PARAM)
    if isinstance(value, str) and re.fullmatch(r"sh_[A-Za-z0-9_-]{24,96}", value):
        return value
    return None


def get_query_value(name: str) -> str | None:
    try:
        value = st.query_params.get(name)
    except Exception:
        return None

    if isinstance(value, list):
        value = value[0] if value else None
    return value if isinstance(value, str) and value else None


def set_query_session_key(session_key: str) -> None:
    try:
        st.query_params[SESSION_QUERY_PARAM] = session_key
    except Exception:
        return


def build_share_url(share_token: str) -> str:
    return f"{read_app_base_url()}/?{urlencode({SHARE_QUERY_PARAM: share_token})}"


def clear_oauth_query_params(session_key: str) -> None:
    try:
        st.query_params.clear()
        st.query_params[SESSION_QUERY_PARAM] = session_key
    except Exception:
        return


def default_greeting() -> dict[str, str]:
    return {
        "role": "assistant",
        "content": (
            "안녕하세요. 오늘 만들고 싶은 변화나 고민을 말해 주세요. "
            "목표를 작게 쪼개서 바로 실행할 수 있게 도와드릴게요."
        ),
    }


def new_conversation_greeting() -> dict[str, str]:
    return {
        "role": "assistant",
        "content": "새 대화를 시작했어요. 지금 가장 바꾸고 싶은 습관부터 말해 주세요.",
    }


def current_auth_user() -> dict[str, str] | None:
    user = st.session_state.get("auth_user")
    return user if isinstance(user, dict) else None


def current_auth_user_id() -> str | None:
    user = current_auth_user()
    if not user:
        return None
    user_id = user.get("id")
    return str(user_id) if user_id else None


def make_pkce_code_verifier() -> str:
    return token_secrets.token_urlsafe(64)


def make_pkce_code_challenge(code_verifier: str) -> str:
    digest = hashlib.sha256(code_verifier.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def normalize_auth_user(user: dict[str, object]) -> dict[str, str]:
    metadata = user.get("user_metadata") if isinstance(user, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}

    identities = user.get("identities") if isinstance(user, dict) else []
    google_sub = metadata.get("sub")
    if not google_sub and isinstance(identities, list):
        for identity in identities:
            if not isinstance(identity, dict):
                continue
            if identity.get("provider") != "google":
                continue
            identity_data = identity.get("identity_data")
            if isinstance(identity_data, dict):
                google_sub = identity_data.get("sub")
            google_sub = google_sub or identity.get("id")
            break

    email = str(user.get("email") or "") if isinstance(user, dict) else ""
    full_name = (
        metadata.get("full_name")
        or metadata.get("name")
        or metadata.get("preferred_username")
        or email
    )

    return {
        "id": str(user.get("id") or ""),
        "email": email,
        "name": str(full_name or email or "Google user"),
        "google_sub": str(google_sub or ""),
    }


def store_auth_response(auth_response: dict[str, object]) -> dict[str, str]:
    auth_user = normalize_auth_user(auth_response["user"])
    if not auth_user.get("id"):
        raise RuntimeError("Supabase user missing")

    st.session_state.auth_user = auth_user
    st.session_state.auth_access_token = str(auth_response.get("access_token") or "")
    st.session_state.auth_refresh_token = str(auth_response.get("refresh_token") or "")
    return auth_user


def restore_auth_session_if_possible() -> None:
    if st.session_state.get("pending_auth_cookie_delete"):
        return
    if current_auth_user():
        return

    refresh_token = st.session_state.get("auth_refresh_token")
    app_auth_token = st.session_state.get("auth_cookie_token") or get_auth_cookie_token()
    if not refresh_token and app_auth_token:
        try:
            app_session = supabase_load_app_auth_session(str(app_auth_token))
            if app_session:
                refresh_token = str(app_session.get("refresh_token") or "")
                st.session_state.auth_cookie_token = str(app_auth_token)
        except Exception:
            refresh_token = None

    if not refresh_token:
        return

    try:
        auth_response = supabase_refresh_auth_session(str(refresh_token))
        auth_user = store_auth_response(auth_response)
        if app_auth_token and auth_response.get("refresh_token"):
            supabase_update_app_auth_session(
                str(app_auth_token),
                auth_user["id"],
                str(auth_response.get("refresh_token") or ""),
            )
        session_key = st.session_state.get("chat_session_key")
        if session_key:
            try:
                restored_messages = supabase_load_messages(str(session_key))
                if restored_messages:
                    st.session_state.messages = restored_messages
            except Exception:
                pass
        st.session_state.auth_status = "Google 로그인: 복원됨"
    except Exception:
        if app_auth_token:
            supabase_revoke_app_auth_session(str(app_auth_token))
            st.session_state.pending_auth_cookie_delete = True
        for key in ("auth_access_token", "auth_refresh_token", "auth_user"):
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.auth_status = "Google 로그인 세션 만료"


def supabase_store_oauth_state(chat_session_key: str, code_verifier: str) -> str:
    state = token_secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(
        minutes=OAUTH_STATE_TTL_MINUTES
    )
    supabase_request(
        "POST",
        "life_coach_oauth_states",
        {
            "state": state,
            "chat_session_key": chat_session_key,
            "code_verifier": code_verifier,
            "expires_at": expires_at.isoformat(),
        },
        prefer="return=minimal",
    )
    return state


def supabase_load_oauth_state(
    state: str | None,
    chat_session_key: str | None,
) -> dict[str, object] | None:
    select_columns = "state,chat_session_key,code_verifier,expires_at"
    now_filter = datetime.now(timezone.utc).isoformat()
    if state:
        params = urlencode(
            {
                "select": select_columns,
                "state": f"eq.{state}",
                "expires_at": f"gt.{now_filter}",
                "limit": "1",
            }
        )
        response = supabase_request("GET", f"life_coach_oauth_states?{params}")
        if isinstance(response, list) and response:
            return response[0]

    if not chat_session_key:
        return None

    params = urlencode(
        {
            "select": select_columns,
            "chat_session_key": f"eq.{chat_session_key}",
            "expires_at": f"gt.{now_filter}",
            "order": "created_at.desc",
            "limit": "1",
        }
    )
    response = supabase_request("GET", f"life_coach_oauth_states?{params}")
    if isinstance(response, list) and response:
        return response[0]
    return None


def supabase_delete_oauth_states(
    chat_session_key: str | None,
    state: str | None = None,
) -> None:
    if state:
        supabase_request(
            "DELETE",
            f"life_coach_oauth_states?state=eq.{quote_plus(state)}",
            prefer="return=minimal",
        )

    if chat_session_key:
        supabase_request(
            "DELETE",
            f"life_coach_oauth_states?chat_session_key=eq.{quote_plus(chat_session_key)}",
            prefer="return=minimal",
        )


def clear_cached_google_oauth_url() -> None:
    for key in (
        "google_oauth_url",
        "google_oauth_session_key",
        "google_oauth_url_created_at",
        "google_oauth_url_version",
    ):
        if key in st.session_state:
            del st.session_state[key]


def render_auth_cookie_scripts() -> None:
    pending_token = st.session_state.get("pending_auth_cookie_token")
    pending_delete = st.session_state.get("pending_auth_cookie_delete")
    secure_attr = "; Secure" if app_runs_on_https() else ""

    if pending_token:
        safe_token = html.escape(str(pending_token), quote=True)
        components.html(
            f"""
<script>
document.cookie = "{AUTH_COOKIE_NAME}={safe_token}; Max-Age={AUTH_SESSION_DAYS * 86400}; Path=/; SameSite=Lax{secure_attr}";
</script>
""",
            height=0,
        )
        st.session_state.auth_cookie_token = str(pending_token)
        del st.session_state.pending_auth_cookie_token

    if pending_delete:
        components.html(
            f"""
<script>
document.cookie = "{AUTH_COOKIE_NAME}=; Max-Age=0; Path=/; SameSite=Lax{secure_attr}";
</script>
""",
            height=0,
        )
        del st.session_state.pending_auth_cookie_delete


def render_browser_head_tags() -> None:
    components.html(
        """
<script>
(function () {
  const appName = "Life Coach";
  const iconHref = "/app/static/icons/icon-192.png";

  const getDocument = (frameWindow) => {
    try {
      return frameWindow && frameWindow.document;
    } catch (error) {
      return null;
    }
  };

  const candidateDocuments = [getDocument(window.parent), getDocument(window.top)].filter(Boolean);

  const updateDocument = (doc) => {
    doc.title = appName;

    doc
      .querySelectorAll('link[rel="manifest"], link[rel="icon"], link[rel="alternate icon"], link[rel="apple-touch-icon"]')
      .forEach((el) => el.remove());

    const upsertLink = (rel, href, attrs = {}) => {
      let el = doc.querySelector(`link[rel="${rel}"][href="${href}"]`);
      if (!el) {
        el = doc.createElement("link");
        el.setAttribute("rel", rel);
        el.setAttribute("href", href);
        doc.head.appendChild(el);
      }
      Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
    };

    const upsertMeta = (name, content) => {
      let el = doc.querySelector(`meta[name="${name}"]`);
      if (!el) {
        el = doc.createElement("meta");
        el.setAttribute("name", name);
        doc.head.appendChild(el);
      }
      el.setAttribute("content", content);
    };

    const upsertStyle = (id, css) => {
      let el = doc.getElementById(id);
      if (!el) {
        el = doc.createElement("style");
        el.setAttribute("id", id);
        doc.head.appendChild(el);
      }
      el.textContent = css;
    };

    upsertLink("icon", iconHref, { sizes: "192x192", type: "image/png" });
    upsertMeta("theme-color", "#2563eb");
    upsertStyle(
      "life-coach-cloud-chrome-hide",
      `
        a[href*="streamlit.io/cloud"],
        a[href*="share.streamlit.io/user/"],
        a[href*="github.com/twsftrp-arch/life-coach-agent"],
        [class*="viewerBadge"],
        [class*="ViewerBadge"],
        [data-testid="stToolbar"] [data-testid="stBaseButton-header"] {
          display: none !important;
          visibility: hidden !important;
          pointer-events: none !important;
        }
      `
    );
  };

  candidateDocuments.forEach((doc) => {
    try {
      updateDocument(doc);
    } catch (error) {
      // Streamlit Cloud may wrap the app in a parent document; ignore inaccessible frames.
    }
  });
})();
</script>
""",
        height=0,
    )


def build_google_oauth_url() -> str | None:
    config = read_supabase_public_config()
    if not config:
        return None

    chat_session_key = str(st.session_state.get("chat_session_key") or "")
    if not chat_session_key:
        return None

    cached = st.session_state.get("google_oauth_url")
    cached_session_key = st.session_state.get("google_oauth_session_key")
    cached_at = float(st.session_state.get("google_oauth_url_created_at") or 0)
    cached_version = st.session_state.get("google_oauth_url_version")
    if (
        cached
        and cached_session_key == chat_session_key
        and cached_version == OAUTH_URL_CACHE_VERSION
        and time.time() - cached_at < (OAUTH_STATE_TTL_MINUTES - 1) * 60
    ):
        return str(cached)

    code_verifier = make_pkce_code_verifier()
    code_challenge = make_pkce_code_challenge(code_verifier)
    supabase_store_oauth_state(chat_session_key, code_verifier)
    redirect_to = (
        f"{read_app_base_url()}/?"
        f"{urlencode({SESSION_QUERY_PARAM: chat_session_key, AUTH_CALLBACK_QUERY_PARAM: 'callback'})}"
    )
    params = urlencode(
        {
            "provider": "google",
            "redirect_to": redirect_to,
            "code_challenge": code_challenge,
            "code_challenge_method": "s256",
        }
    )
    login_url = f"{config['url']}/auth/v1/authorize?{params}"
    st.session_state.google_oauth_url = login_url
    st.session_state.google_oauth_session_key = chat_session_key
    st.session_state.google_oauth_url_created_at = time.time()
    st.session_state.google_oauth_url_version = OAUTH_URL_CACHE_VERSION
    return login_url


def exchange_google_oauth_code(auth_code: str, code_verifier: str) -> dict[str, object]:
    response = supabase_auth_request(
        "POST",
        "token?grant_type=pkce",
        {
            "auth_code": auth_code,
            "code_verifier": code_verifier,
        },
    )
    if not isinstance(response, dict) or not response.get("user"):
        raise RuntimeError("Supabase Auth session missing")
    return response


def handle_google_oauth_callback() -> bool:
    auth_code = get_query_value("code")
    auth_error = get_query_value("error") or get_query_value("error_code")
    chat_session_key = get_query_session_key() or str(
        st.session_state.get("chat_session_key") or ""
    )
    if not auth_code and auth_error:
        error_code = get_query_value("error_code") or auth_error
        error_description = get_query_value("error_description") or ""
        if error_code == "bad_oauth_state":
            st.session_state.auth_status = (
                "Google 로그인 링크를 갱신했어요. 다시 로그인해 주세요."
            )
        else:
            st.session_state.auth_status = (
                f"Google 로그인 실패: {error_code}"
                + (f" ({error_description})" if error_description else "")
            )
        clear_cached_google_oauth_url()
        if chat_session_key:
            clear_oauth_query_params(chat_session_key)
        return True

    if not auth_code:
        return False

    state = get_query_value("state")
    try:
        state_row = supabase_load_oauth_state(state, chat_session_key)
        if not state_row:
            raise RuntimeError("OAuth state expired")

        state_session_key = str(state_row.get("chat_session_key") or chat_session_key)
        auth_response = exchange_google_oauth_code(
            auth_code,
            str(state_row.get("code_verifier") or ""),
        )
        auth_user = store_auth_response(auth_response)
        refresh_token = str(auth_response.get("refresh_token") or "")
        if refresh_token:
            app_auth_token = supabase_create_app_auth_session(
                auth_user["id"],
                refresh_token,
            )
            st.session_state.pending_auth_cookie_token = app_auth_token
        st.session_state.auth_status = "Google 로그인: 연결됨"
        st.session_state.chat_session_key = state_session_key
        set_query_session_key(state_session_key)
        supabase_attach_session_to_user(state_session_key, auth_user["id"])
        try:
            restored_messages = supabase_load_messages(state_session_key)
            if restored_messages:
                st.session_state.messages = restored_messages
        except Exception:
            pass
        supabase_delete_oauth_states(state_session_key, state)
        clear_cached_google_oauth_url()
        clear_oauth_query_params(state_session_key)
        return True
    except Exception as exc:
        st.session_state.auth_status = f"Google 로그인 실패: {exc.__class__.__name__}"
        clear_cached_google_oauth_url()
        if chat_session_key:
            clear_oauth_query_params(chat_session_key)
        return True


def supabase_get_session_row(session_key: str) -> dict[str, object] | None:
    response = supabase_request(
        "GET",
        f"life_coach_sessions?select=id,title,user_id&session_key=eq.{quote_plus(session_key)}&limit=1",
    )
    if isinstance(response, list) and response:
        item = response[0]
        return item if isinstance(item, dict) else None
    return None


def ensure_session_owner_access(session_row: dict[str, object] | None) -> None:
    if not session_row:
        return

    owner_id = session_row.get("user_id")
    if not owner_id:
        return

    current_user_id = current_auth_user_id()
    if str(owner_id) != str(current_user_id or ""):
        raise PermissionError("Supabase session owner mismatch")


def supabase_ensure_session(session_key: str, title: str | None = None) -> str:
    existing_row = supabase_get_session_row(session_key)
    ensure_session_owner_access(existing_row)

    payload: dict[str, object] = {"session_key": session_key}
    user_id = current_auth_user_id()
    existing_owner_id = str(existing_row.get("user_id") or "") if existing_row else ""
    if user_id and not existing_owner_id:
        payload["user_id"] = user_id
    if title:
        has_title = bool(str(existing_row.get("title") or "").strip()) if existing_row else False
        if not has_title:
            payload["title"] = title[:120]

    params = urlencode({"on_conflict": "session_key"})
    response = supabase_request(
        "POST",
        f"life_coach_sessions?{params}",
        payload,
        prefer="resolution=merge-duplicates,return=representation",
    )
    if isinstance(response, list) and response:
        session_id = response[0].get("id")
    elif isinstance(response, dict):
        session_id = response.get("id")
    else:
        session_id = None

    if not session_id:
        raise RuntimeError("Supabase session id missing")
    return str(session_id)


def supabase_attach_session_to_user(session_key: str, user_id: str) -> None:
    supabase_ensure_session(session_key)
    supabase_request(
        "PATCH",
        f"life_coach_sessions?session_key=eq.{quote_plus(session_key)}",
        {"user_id": user_id},
        prefer="return=minimal",
    )


def supabase_list_user_sessions(user_id: str, limit: int = 20) -> list[dict[str, object]]:
    params = urlencode(
        {
            "select": "session_key,title,created_at,updated_at",
            "user_id": f"eq.{user_id}",
            "order": "updated_at.desc",
            "limit": str(limit),
        }
    )
    response = supabase_request("GET", f"life_coach_sessions?{params}")
    if not isinstance(response, list):
        return []
    return [item for item in response if isinstance(item, dict)]


def supabase_update_session_title(
    session_key: str,
    user_id: str,
    title: str,
) -> None:
    if not user_id:
        raise PermissionError("login required")
    clean_title = " ".join(title.split()).strip()[:120]
    if not clean_title:
        raise ValueError("empty title")

    params = urlencode(
        {
            "session_key": f"eq.{session_key}",
            "user_id": f"eq.{user_id}",
        }
    )
    supabase_request(
        "PATCH",
        f"life_coach_sessions?{params}",
        {"title": clean_title},
        prefer="return=minimal",
    )


def supabase_delete_session(session_key: str, user_id: str) -> None:
    if not user_id:
        raise PermissionError("login required")
    session_params = urlencode(
        {
            "select": "id",
            "session_key": f"eq.{session_key}",
            "user_id": f"eq.{user_id}",
            "limit": "1",
        }
    )
    session_response = supabase_request("GET", f"life_coach_sessions?{session_params}")
    if not isinstance(session_response, list) or not session_response:
        raise RuntimeError("Supabase session not found")

    session_id = str(session_response[0].get("id") or "")
    if not session_id:
        raise RuntimeError("Supabase session id missing")

    supabase_request(
        "DELETE",
        f"life_coach_messages?session_id=eq.{quote_plus(session_id)}",
        prefer="return=minimal",
    )
    supabase_request(
        "DELETE",
        f"life_coach_sessions?id=eq.{quote_plus(session_id)}&user_id=eq.{quote_plus(user_id)}",
        prefer="return=minimal",
    )


def format_saved_session_label(item: dict[str, object]) -> str:
    title = str(item.get("title") or "").strip()
    if not title:
        title = "제목 없는 대화"
    if len(title) > 34:
        title = f"{title[:31]}..."
    updated_at = str(item.get("updated_at") or item.get("created_at") or "")
    date_label = updated_at[:10] if updated_at else ""
    return f"{title} · {date_label}" if date_label else title


def supabase_load_messages(session_key: str) -> list[dict[str, object]]:
    filter_value = quote_plus(session_key)
    session_response = supabase_request(
        "GET",
        f"life_coach_sessions?select=id,user_id&session_key=eq.{filter_value}&limit=1",
    )
    if not isinstance(session_response, list) or not session_response:
        return []

    ensure_session_owner_access(session_response[0])

    session_id = session_response[0].get("id")
    if not session_id:
        return []

    message_params = urlencode(
        {
            "select": "role,content,evidence",
            "session_id": f"eq.{session_id}",
            "order": "created_at.asc,id.asc",
        }
    )
    message_response = supabase_request("GET", f"life_coach_messages?{message_params}")
    if not isinstance(message_response, list):
        return []

    messages: list[dict[str, object]] = []
    for item in message_response:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        message: dict[str, object] = {"role": role, "content": content}
        if item.get("evidence"):
            message["evidence"] = item.get("evidence")
        messages.append(message)

    return messages


def switch_conversation(session_key: str) -> None:
    st.session_state.chat_session_key = session_key
    set_query_session_key(session_key)
    st.session_state.session_id = str(session_key)
    st.session_state.agent_session = SQLiteSession(
        st.session_state.session_id,
        str(DB_PATH),
    )
    try:
        messages = supabase_load_messages(session_key)
        st.session_state.supabase_status = f"Supabase 복원: {len(messages)}개 메시지"
    except Exception as exc:
        messages = []
        st.session_state.supabase_status = f"Supabase 복원 실패: {exc.__class__.__name__}"

    st.session_state.messages = messages or [default_greeting()]


def persist_chat_message(
    role: str,
    content: str,
    evidence: dict[str, object] | None = None,
) -> None:
    if role not in {"user", "assistant"}:
        return

    session_key = st.session_state.get("chat_session_key")
    if not session_key:
        return

    try:
        session_id = supabase_ensure_session(
            str(session_key),
            title=content if role == "user" else None,
        )
        safe_evidence = None
        if evidence is not None:
            safe_evidence = json.loads(json.dumps(evidence, default=str))
        supabase_request(
            "POST",
            "life_coach_messages",
            {
                "session_id": session_id,
                "role": role,
                "content": content,
                "evidence": safe_evidence,
            },
            prefer="return=minimal",
        )
        st.session_state.supabase_status = "Supabase 저장: 연결됨"
    except Exception as exc:
        st.session_state.supabase_status = f"Supabase 저장 실패: {exc.__class__.__name__}"


def make_share_token() -> str:
    return f"sh_{token_secrets.token_urlsafe(32)}"


def sanitize_messages_for_share(
    messages: list[dict[str, object]],
) -> list[dict[str, str]]:
    shared_messages: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        clean_content = content.strip()
        if not clean_content:
            continue
        shared_messages.append({"role": str(role), "content": clean_content})
    return shared_messages


def title_from_messages(messages: list[dict[str, str]]) -> str:
    for message in messages:
        if message.get("role") != "user":
            continue
        content = " ".join(str(message.get("content") or "").split())
        if content:
            return content[:80]
    return "공유된 Life Coach 대화"


def supabase_create_shared_chat(
    session_key: str,
    user_id: str,
    messages: list[dict[str, object]],
) -> dict[str, str]:
    if not user_id:
        raise PermissionError("login required")

    shared_messages = sanitize_messages_for_share(messages)
    if not any(message["role"] == "user" for message in shared_messages):
        raise ValueError("share requires a user message")

    session_id = supabase_ensure_session(
        session_key,
        title=title_from_messages(shared_messages),
    )
    session_row = supabase_get_session_row(session_key)
    ensure_session_owner_access(session_row)

    share_token = make_share_token()
    title = title_from_messages(shared_messages)
    response = supabase_request(
        "POST",
        "life_coach_shared_chats",
        {
            "share_token": share_token,
            "source_session_id": session_id,
            "source_session_key": session_key,
            "owner_user_id": user_id,
            "title": title,
            "messages": shared_messages,
        },
        prefer="return=representation",
    )
    if isinstance(response, list) and response:
        item = response[0]
        if isinstance(item, dict):
            token = str(item.get("share_token") or share_token)
            return {"share_token": token, "url": build_share_url(token)}
    return {"share_token": share_token, "url": build_share_url(share_token)}


def supabase_list_shared_chats(
    session_key: str,
    user_id: str,
) -> list[dict[str, object]]:
    if not user_id:
        return []
    params = urlencode(
        {
            "select": "share_token,title,created_at,revoked_at",
            "source_session_key": f"eq.{session_key}",
            "owner_user_id": f"eq.{user_id}",
            "revoked_at": "is.null",
            "order": "created_at.desc",
            "limit": "5",
        }
    )
    response = supabase_request("GET", f"life_coach_shared_chats?{params}")
    if not isinstance(response, list):
        return []
    return [item for item in response if isinstance(item, dict)]


def supabase_load_shared_chat(share_token: str) -> dict[str, object] | None:
    params = urlencode(
        {
            "select": "share_token,title,messages,created_at,revoked_at",
            "share_token": f"eq.{share_token}",
            "revoked_at": "is.null",
            "limit": "1",
        }
    )
    response = supabase_request("GET", f"life_coach_shared_chats?{params}")
    if isinstance(response, list) and response:
        item = response[0]
        return item if isinstance(item, dict) else None
    return None


def supabase_revoke_shared_chat(share_token: str, user_id: str) -> None:
    if not user_id:
        raise PermissionError("login required")
    params = urlencode(
        {
            "share_token": f"eq.{share_token}",
            "owner_user_id": f"eq.{user_id}",
        }
    )
    supabase_request(
        "PATCH",
        f"life_coach_shared_chats?{params}",
        {"revoked_at": datetime.now(timezone.utc).isoformat()},
        prefer="return=minimal",
    )


def supabase_load_user_preferences(user_id: str) -> dict[str, str] | None:
    if not user_id:
        return None
    params = urlencode(
        {
            "select": "coaching_style,custom_instructions",
            "user_id": f"eq.{user_id}",
            "limit": "1",
        }
    )
    response = supabase_request("GET", f"life_coach_user_preferences?{params}")
    if not isinstance(response, list) or not response:
        return None
    item = response[0]
    if not isinstance(item, dict):
        return None
    return {
        "coaching_style": normalize_coaching_style(str(item.get("coaching_style") or "")),
        "custom_instructions": clean_custom_instructions(
            item.get("custom_instructions")
        ),
    }


def supabase_upsert_user_preferences(
    user_id: str,
    coaching_style: str,
    custom_instructions: str,
) -> None:
    if not user_id:
        raise PermissionError("login required")
    payload = {
        "user_id": user_id,
        "coaching_style": normalize_coaching_style(coaching_style),
        "custom_instructions": clean_custom_instructions(custom_instructions),
    }
    response = supabase_request(
        "POST",
        "life_coach_user_preferences?on_conflict=user_id",
        payload,
        prefer="resolution=merge-duplicates,return=minimal",
    )
    return None


def restore_user_preferences_if_possible() -> None:
    user_id = current_auth_user_id()
    if not user_id:
        return
    if st.session_state.get("preferences_loaded_for_user") == user_id:
        return

    try:
        preferences = supabase_load_user_preferences(user_id)
        if preferences:
            st.session_state.coaching_style = normalize_coaching_style(
                preferences.get("coaching_style")
            )
            st.session_state.custom_coach_instructions = clean_custom_instructions(
                preferences.get("custom_instructions")
            )
            st.session_state.preference_status = "코칭 설정: 복원됨"
        else:
            st.session_state.preference_status = "코칭 설정: 기본값"
        st.session_state.preferences_loaded_for_user = user_id
    except Exception as exc:
        st.session_state.preference_status = (
            f"코칭 설정 복원 실패: {exc.__class__.__name__}"
        )


def search_web_raw(query: str) -> str:
    """Search the public web and return compact text results."""
    clean_query = query.strip()
    if not clean_query:
        return "검색어가 비어 있습니다."

    url = f"https://duckduckgo.com/html/?q={quote_plus(clean_query)}"
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
            )
        },
    )

    try:
        with urlopen(request, timeout=10) as response:
            html = response.read().decode("utf-8", errors="replace")
    except (OSError, URLError) as exc:
        return f"웹 검색 중 오류가 발생했습니다: {exc.__class__.__name__}"

    parser = SearchResultParser()
    parser.feed(html)
    parser.close()

    if not parser.results:
        return "검색 결과를 찾지 못했습니다. 검색어를 더 구체적으로 바꿔 보세요."

    lines = []
    for index, result in enumerate(parser.results[:5], start=1):
        snippet = result["snippet"] or "요약 없음"
        lines.append(
            f"{index}. {result['title']}\n"
            f"URL: {result['url']}\n"
            f"요약: {snippet}"
        )

    return "\n\n".join(lines)


def extract_result_urls(text: str) -> list[str]:
    return re.findall(r"^URL:\s*(\S+)", text, flags=re.MULTILINE)


def append_run_event(message: str) -> None:
    events = RUN_EVENTS.get()
    if events is None:
        return

    started = RUN_STARTED_AT.get()
    seconds = 0.0 if started is None else time.perf_counter() - started
    events.append({"seconds": seconds, "message": message})

    renderer = RUN_EVENT_RENDERER.get()
    event_queue = RUN_EVENT_QUEUE.get()
    if event_queue:
        event_queue.put(list(events))
    elif renderer:
        renderer(events)


def format_run_events_markdown(
    events: list[dict[str, object]],
    active_message: str | None = None,
    active_seconds: float | None = None,
    title: str = "실시간 실행 로그",
) -> str:
    lines = [f"**{title}**"]
    previous_seconds = 0.0
    for event in events:
        seconds = float(event.get("seconds") or 0)
        message = event.get("message", "")
        interval_seconds = max(0.0, seconds - previous_seconds)
        previous_seconds = seconds
        lines.append(
            f"- `+{format_seconds(interval_seconds)}` "
            f"`t+{format_seconds(seconds)}` {message}"
        )

    if active_message and active_seconds is not None:
        lines.append(f"- 진행 중: {active_message} `{format_seconds(active_seconds)}`")

    return "\n".join(lines)


@function_tool
def search_web(query: str) -> str:
    """Search the web for motivation, self-development, and habit-building advice."""
    search_call_count = SEARCH_CALL_COUNT.get()
    if search_call_count is not None:
        if search_call_count[0] >= MAX_SEARCH_CALLS_PER_MESSAGE:
            append_run_event(
                f"`search_web` tool 추가 호출 차단: {query} "
                f"(이번 메시지 최대 {MAX_SEARCH_CALLS_PER_MESSAGE}회)"
            )
            return (
                "이미 이번 사용자 메시지에서 충분한 웹 검색을 수행했습니다. "
                "추가 검색 없이 앞선 검색 결과를 바탕으로 답변하세요."
            )
        search_call_count[0] += 1

    started = time.perf_counter()
    append_run_event(f"`search_web` tool 호출: {query}")
    output = search_web_raw(query)
    elapsed = time.perf_counter() - started
    append_run_event(f"`search_web` tool 완료: {format_seconds(elapsed)}")
    append_run_event("모델 답변 생성 대기 시작")

    timings = SEARCH_TIMINGS.get()
    if timings is not None:
        timings.append(
            {
                "query": query,
                "seconds": elapsed,
                "urls": extract_result_urls(output)[:3],
                "output": output,
            }
        )

    return output


def _split_goal_chunks(text: str) -> list[str]:
    """Split a goal/journal document into heading-based sections."""
    chunks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.lstrip().startswith("#") and any(c.strip() for c in current):
            chunks.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        chunks.append("\n".join(current).strip())
    return [chunk for chunk in chunks if chunk.strip()]


def search_goals_in_text(goals_text: str, query: str) -> str:
    """Return the goal/journal sections most relevant to the query."""
    clean = (goals_text or "").strip()
    if not clean:
        return (
            "업로드된 개인 목표 문서가 없습니다. "
            "사이드바에서 목표 파일을 올리면 검색할 수 있어요."
        )

    normalized_query = " ".join(query.split()).lower()
    tokens = [token for token in normalized_query.split() if len(token) >= 2]

    scored: list[tuple[int, str]] = []
    for chunk in _split_goal_chunks(clean):
        low = chunk.lower()
        score = sum(low.count(token) for token in tokens)
        if normalized_query and normalized_query in low:
            score += 3
        if score > 0:
            scored.append((score, chunk))

    if not scored:
        return (
            "질문과 정확히 일치하는 항목은 찾지 못했습니다. "
            "참고용으로 목표 문서 일부를 제공합니다:\n\n" + clean[:1500]
        )

    scored.sort(key=lambda item: item[0], reverse=True)
    top_chunks = [chunk for _, chunk in scored[:3]]
    return "\n\n---\n\n".join(top_chunks)[:4000]


def make_search_goals_tool(goals_text: str):
    """Build a file-search tool bound to the current user's goal document."""

    @function_tool
    def search_goals(query: str) -> str:
        """Search the user's personal goals and journal entries for relevant context."""
        call_count = GOAL_SEARCH_CALL_COUNT.get()
        if call_count is not None:
            if call_count[0] >= MAX_GOAL_SEARCH_CALLS_PER_MESSAGE:
                append_run_event(f"`search_goals` tool 추가 호출 차단: {query}")
                return (
                    "이미 개인 목표 문서를 충분히 확인했습니다. "
                    "앞선 목표/기록 내용을 바탕으로 답변하세요."
                )
            call_count[0] += 1

        started = time.perf_counter()
        append_run_event(f"`search_goals` tool 호출: {query}")
        output = search_goals_in_text(goals_text, query)
        elapsed = time.perf_counter() - started
        append_run_event(f"`search_goals` tool 완료: {format_seconds(elapsed)}")

        timings = GOALS_TIMINGS.get()
        if timings is not None:
            timings.append(
                {
                    "query": query,
                    "seconds": elapsed,
                    "output": output,
                }
            )
        return output

    return search_goals


def format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    return f"{seconds:.2f}s"


def format_clock_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "0.0s"
    return f"{max(0.0, seconds):.1f}s"


def format_shared_timestamp(value: object) -> str:
    raw_value = str(value or "").strip()
    if not raw_value:
        return ""

    try:
        parsed = datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
    except ValueError:
        return raw_value[:16].replace("T", " ")

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(KST).strftime("%Y-%m-%d %H:%M KST")


def linkify_plain_urls(text: str) -> str:
    url_pattern = re.compile(r"(?<![\]\(<])https?://[^\s<>)]+")

    def replace(match: re.Match[str]) -> str:
        raw_url = match.group(0)
        url = raw_url.rstrip(".,;:")
        suffix = raw_url[len(url) :]
        return f"<{url}>{suffix}"

    return url_pattern.sub(replace, text)


def format_markdown_url(url: object) -> str:
    clean_url = str(url).strip()
    if not clean_url:
        return ""

    safe_url = (
        clean_url.replace(" ", "%20")
        .replace("<", "%3C")
        .replace(">", "%3E")
    )
    parsed = urlparse(safe_url)
    label = parsed.netloc or safe_url
    path = unquote(parsed.path).rstrip("/")
    if path and path != "/":
        leaf = path.rsplit("/", 1)[-1] or path
        if len(leaf) > 36:
            leaf = f"{leaf[:33]}..."
        label = f"{label}/{leaf}"

    label = label.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
    return f"[{label}](<{safe_url}>)"


def normalize_thinking_mode(thinking_mode: str | None) -> str:
    if thinking_mode in THINKING_MODES:
        return str(thinking_mode)
    return DEFAULT_THINKING_MODE


def thinking_mode_label(thinking_mode: str | None) -> str:
    mode = normalize_thinking_mode(thinking_mode)
    return str(THINKING_MODES[mode]["label"])


def normalize_coaching_style(style: str | None) -> str:
    if style in COACHING_STYLES:
        return str(style)
    return DEFAULT_COACHING_STYLE


def coaching_style_label(style: object) -> str:
    normalized = normalize_coaching_style(str(style))
    return COACHING_STYLES[normalized]["label"]


def coaching_style_description(style: str | None) -> str:
    normalized = normalize_coaching_style(style)
    return COACHING_STYLES[normalized]["description"]


def clean_custom_instructions(value: object) -> str:
    text = str(value or "").replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:CUSTOM_INSTRUCTIONS_MAX_CHARS]


def current_coach_preferences() -> dict[str, str]:
    style = normalize_coaching_style(st.session_state.get("coaching_style"))
    custom = clean_custom_instructions(
        st.session_state.get("custom_coach_instructions")
    )
    return {"coaching_style": style, "custom_instructions": custom}


def compose_coach_instructions(base_instructions: str) -> str:
    preferences = current_coach_preferences()
    style = normalize_coaching_style(preferences.get("coaching_style"))
    custom = clean_custom_instructions(preferences.get("custom_instructions"))
    style_config = COACHING_STYLES[style]

    preference_lines = [
        "User-selected coaching preferences:",
        f"- Coaching style: {style_config['label']}",
        f"- Style instruction: {style_config['instructions']}",
    ]
    if custom:
        preference_lines.extend(
            [
                "- User custom coaching instruction:",
                custom,
            ]
        )
    preference_lines.append(
        "These preferences are lower priority than safety, medical/crisis "
        "guidance, source-formatting rules, and the core life-coach role."
    )

    return f"{base_instructions.strip()}\n\n" + "\n".join(preference_lines)


def model_label(model: object) -> str:
    return MODEL_LABELS.get(str(model), "Model")


def build_model_settings(thinking_mode: str | None) -> ModelSettings:
    mode = normalize_thinking_mode(thinking_mode)
    config = THINKING_MODES[mode]
    thinking_type = config["thinking_type"]
    effort = config["effort"]

    return ModelSettings(
        parallel_tool_calls=False,
        extra_body={"thinking": {"type": thinking_type}},
        reasoning=Reasoning(effort=effort) if effort else None,
    )


def attach_runtime_settings(
    evidence: dict[str, object],
    model: str,
    thinking_mode: str,
) -> dict[str, object]:
    evidence["model"] = evidence.get("model") or model
    evidence["thinking_mode"] = thinking_mode_label(thinking_mode)
    return evidence


def render_status_message(
    placeholder,
    message: str,
    seconds: float | None = None,
) -> None:
    time_text = format_clock_seconds(seconds)
    placeholder.markdown(
        f"""
<div class="run-status-box">
  <span class="run-status-dot"></span>
  <span class="run-status-text">{message}</span>
  <code>{time_text}</code>
</div>
""",
        unsafe_allow_html=True,
    )


def copy_text_to_clipboard(text: str, label: str) -> None:
    if shutil.which("pbcopy") is None:
        st.session_state.copy_notice = (
            f"{label}: 배포 환경에서는 자동 복사가 제한되어 아래 텍스트를 직접 복사하세요."
        )
        st.session_state.copy_fallback_text = text
        return

    try:
        subprocess.run(
            ["pbcopy"],
            input=text,
            text=True,
            check=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        st.session_state.copy_notice = f"{label} 실패: {exc.__class__.__name__}"
        return

    st.session_state.copy_notice = f"{label} 완료"


def render_copy_button(text: str, button_id: str, label: str) -> None:
    safe_key = re.sub(r"[^a-zA-Z0-9_-]", "-", button_id)
    st.button(
        label,
        key=f"copy-button-{safe_key}",
        on_click=copy_text_to_clipboard,
        args=(text, label),
    )


def render_copy_feedback() -> None:
    copy_notice = st.session_state.get("copy_notice")
    if copy_notice:
        st.toast(copy_notice)
        del st.session_state.copy_notice

    copy_fallback_text = st.session_state.get("copy_fallback_text")
    if not copy_fallback_text:
        return

    st.info("자동 복사가 지원되지 않는 환경입니다. 아래 내용을 선택해서 복사하세요.")
    st.text_area(
        "복사할 텍스트",
        value=copy_fallback_text,
        height=160,
        key="copy-fallback-text-area",
    )
    if st.button("복사 안내 닫기", key="close-copy-fallback"):
        del st.session_state.copy_fallback_text
        st.rerun()


def render_web_share_actions(
    share_url: str,
    key_suffix: str,
) -> None:
    payload = {
        "title": APP_TITLE,
        "text": "Life Coach 대화를 공유합니다.",
        "url": share_url,
    }
    payload_json = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    safe_key = re.sub(r"[^a-zA-Z0-9_-]", "-", key_suffix)
    components.html(
        f"""
<div class="share-actions">
  <button id="share-{safe_key}" type="button">공유 앱 선택</button>
  <button id="copy-{safe_key}" type="button">링크 복사</button>
  <span id="status-{safe_key}" aria-live="polite">버튼을 눌러 공유하세요.</span>
</div>
<script>
(function () {{
  const payload = {payload_json};
  const statusEl = document.getElementById("status-{safe_key}");
  const shareButton = document.getElementById("share-{safe_key}");
  const copyButton = document.getElementById("copy-{safe_key}");
  const nav = (() => {{
    try {{
      return window.parent && window.parent.navigator
        ? window.parent.navigator
        : window.navigator;
    }} catch (error) {{
      return window.navigator;
    }}
  }})();

  function setStatus(message) {{
    statusEl.textContent = message;
  }}

  async function copyLink() {{
    try {{
      const clipboard = nav.clipboard || window.navigator.clipboard;
      await clipboard.writeText(payload.url);
      setStatus("링크를 복사했어요.");
    }} catch (error) {{
      setStatus("브라우저에서 직접 링크를 복사해 주세요.");
    }}
  }}

  async function shareLink() {{
    try {{
      if (nav.share) {{
        await nav.share(payload);
        setStatus("공유 창을 열었어요.");
        return;
      }}
      await copyLink();
    }} catch (error) {{
      if (error && error.name === "AbortError") {{
        setStatus("공유를 취소했어요.");
        return;
      }}
      await copyLink();
    }}
  }}

  shareButton.addEventListener("click", shareLink);
  copyButton.addEventListener("click", copyLink);
}})();
</script>
<style>
  .share-actions {{
    align-items: center;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  .share-actions button {{
    background: #111827;
    border: 1px solid #111827;
    border-radius: 6px;
    color: #ffffff;
    cursor: pointer;
    font-size: 14px;
    min-height: 34px;
    padding: 6px 10px;
  }}
  .share-actions button + button {{
    background: #ffffff;
    color: #111827;
  }}
  .share-actions span {{
    color: #4b5563;
    font-size: 13px;
  }}
</style>
""",
        height=64,
    )


def extract_run_evidence(result) -> dict[str, object]:
    evidence: dict[str, object] = {
        "model": None,
        "searches": [],
        "total_seconds": None,
    }
    searches: list[dict[str, object]] = []

    for item in getattr(result, "new_items", []):
        raw_item = getattr(item, "raw_item", None)
        provider_data = getattr(raw_item, "provider_data", None)
        if isinstance(provider_data, dict) and provider_data.get("model"):
            evidence["model"] = provider_data["model"]

        if type(item).__name__ == "ToolCallItem":
            tool_name = getattr(raw_item, "name", "")
            if tool_name != "search_web":
                continue

            raw_arguments = getattr(raw_item, "arguments", "{}")
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                arguments = {}

            searches.append(
                {
                    "query": arguments.get("query", ""),
                    "urls": [],
                }
            )

        if type(item).__name__ == "ToolCallOutputItem":
            output = getattr(item, "output", "")
            urls = extract_result_urls(str(output))
            if searches:
                if "이미 이번 사용자 메시지에서 충분한 웹 검색을 수행했습니다" in str(
                    output
                ):
                    searches[-1]["blocked"] = True
                if urls:
                    searches[-1]["urls"] = urls[:3]

    evidence["searches"] = searches
    return evidence


def merge_search_timings(
    evidence: dict[str, object],
    timings: list[dict[str, object]],
) -> dict[str, object]:
    searches = evidence.get("searches")
    if not isinstance(searches, list):
        searches = []

    for index, timing in enumerate(timings):
        if index >= len(searches) or not isinstance(searches[index], dict):
            searches.append({})

        searches[index].update(
            {
                "query": searches[index].get("query") or timing.get("query"),
                "seconds": timing.get("seconds"),
                "urls": searches[index].get("urls") or timing.get("urls") or [],
            }
        )

    evidence["searches"] = searches
    return evidence


def render_run_evidence(evidence: dict[str, object] | None) -> None:
    if not evidence:
        return

    model = evidence.get("model")
    display_model = model_label(model) if model else None
    thinking_mode = evidence.get("thinking_mode")
    searches = evidence.get("searches") or []
    events = evidence.get("events") or []
    if not model and not searches and not events:
        return

    actual_searches = [
        search
        for search in searches
        if isinstance(search, dict) and not search.get("blocked")
    ]
    blocked_searches = [
        search
        for search in searches
        if isinstance(search, dict) and search.get("blocked")
    ]
    total_tool_seconds = sum(
        float(search.get("seconds") or 0) for search in actual_searches
    )

    if evidence.get("search_then_stream"):
        mode_label = "검색 후 스트리밍"
    elif evidence.get("streaming") is True:
        mode_label = "스트리밍"
    elif evidence.get("streaming") is False:
        mode_label = "동기 실행"
    else:
        mode_label = None

    summary_parts = []
    if display_model:
        summary_parts.append(display_model)
    if mode_label:
        summary_parts.append(mode_label)
    if thinking_mode:
        summary_parts.append(str(thinking_mode))
    if evidence.get("total_seconds") is not None:
        summary_parts.append(f"총 {format_seconds(evidence.get('total_seconds'))}")
    goal_searches = evidence.get("goal_searches")
    if isinstance(goal_searches, list) and goal_searches:
        summary_parts.append(f"개인 목표 검색 {len(goal_searches)}회")
    if actual_searches:
        summary_parts.append(
            f"웹 검색 {len(actual_searches)}회/{format_seconds(total_tool_seconds)}"
        )
    if blocked_searches:
        summary_parts.append(f"추가 검색 차단 {len(blocked_searches)}회")
    if not summary_parts:
        summary_parts.append("상세 로그 저장됨")

    st.caption(f"실행 확인: {' · '.join(summary_parts)}")

    with st.expander("상세 실행 정보", expanded=False):
        detail_lines = ["**요약**"]
        if display_model:
            detail_lines.append(f"- 모델: {display_model}")
        if mode_label:
            detail_lines.append(f"- 응답 방식: {mode_label}")
        if thinking_mode:
            detail_lines.append(f"- 사고 모드: {thinking_mode}")
        if evidence.get("total_seconds") is not None:
            detail_lines.append(
                f"- 총 응답 시간: {format_seconds(evidence.get('total_seconds'))}"
            )
        if evidence.get("fallback_reason"):
            detail_lines.append(
                f"- 스트리밍 fallback: `{evidence.get('fallback_reason')}`"
            )
        if evidence.get("streaming_skip_reason"):
            detail_lines.append(
                f"- 스트리밍 생략: {evidence.get('streaming_skip_reason')}"
            )
        if evidence.get("session_recovered"):
            detail_lines.append("- 세션 복구: BadRequestError 이후 새 SDK 세션으로 재시도")
        if actual_searches:
            detail_lines.append(
                f"- 웹 검색: {len(actual_searches)}회 "
                f"({format_seconds(total_tool_seconds)})"
            )
        if blocked_searches:
            detail_lines.append(f"- 추가 검색 시도 차단: {len(blocked_searches)}회")
        st.markdown("\n".join(detail_lines))

        if actual_searches:
            search_lines = ["**웹 검색**"]
            show_search_numbers = len(actual_searches) > 1
            for index, search in enumerate(actual_searches, start=1):
                query = search.get("query") or "(검색어 없음)"
                seconds = format_seconds(float(search.get("seconds") or 0))
                if show_search_numbers:
                    search_lines.append(f"- 검색 {index}: `{query}` ({seconds})")
                else:
                    search_lines.append(f"- 검색어: `{query}` ({seconds})")
                urls = search.get("urls") or []
                for url in urls:
                    link = format_markdown_url(url) or str(url)
                    search_lines.append(f"  - 출처: {link}")
            st.markdown("\n".join(search_lines))

        if events:
            st.markdown(
                format_run_events_markdown(
                    events,
                    title="완료된 실행 타임라인",
                )
            )


def build_openai_compatible_model(model: str, api_key: str) -> OpenAIChatCompletionsModel:
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
    )

    return OpenAIChatCompletionsModel(
        model=model,
        openai_client=client,
    )


def build_agent(model: str, api_key: str, thinking_mode: str) -> Agent:
    return Agent(
        name="Life Coach",
        model=build_openai_compatible_model(model, api_key),
        instructions=compose_coach_instructions(LIFE_COACH_INSTRUCTIONS),
        model_settings=build_model_settings(thinking_mode),
        tools=[search_web],
    )


def build_search_agent(
    model: str,
    api_key: str,
    thinking_mode: str,
    goals_text: str = "",
) -> Agent:
    tools = [search_web]
    if goals_text and goals_text.strip():
        # search_goals first so the planner recalls personal goals before the web.
        tools = [make_search_goals_tool(goals_text), search_web]
    return Agent(
        name="Life Coach Researcher",
        model=build_openai_compatible_model(model, api_key),
        instructions=SEARCH_AGENT_INSTRUCTIONS,
        model_settings=build_model_settings(thinking_mode),
        tools=tools,
    )


def build_streaming_coach_agent(model: str, api_key: str, thinking_mode: str) -> Agent:
    return Agent(
        name="Life Coach",
        model=build_openai_compatible_model(model, api_key),
        instructions=compose_coach_instructions(STREAMING_COACH_INSTRUCTIONS),
        model_settings=build_model_settings(thinking_mode),
        tools=[],
    )


async def run_agent_streamed(
    agent: Agent,
    prompt: str,
    session: SQLiteSession,
    response_placeholder,
    status_placeholder,
    activity_renderer: Callable[[list[dict[str, object]]], None] | None = None,
    initial_events: list[dict[str, object]] | None = None,
    started_at: float | None = None,
    mode_message: str = "자동 모드: 일반 대화라 스트리밍",
    stop_event: threading.Event | None = None,
) -> tuple[str, dict[str, object]]:
    started = started_at or time.perf_counter()
    search_timings: list[dict[str, object]] = []
    run_events: list[dict[str, object]] = list(initial_events or [])
    timing_token = SEARCH_TIMINGS.set(search_timings)
    events_token = RUN_EVENTS.set(run_events)
    started_token = RUN_STARTED_AT.set(started)
    renderer_token = RUN_EVENT_RENDERER.set(activity_renderer)
    queue_token = RUN_EVENT_QUEUE.set(None)
    search_count_token = SEARCH_CALL_COUNT.set([0])
    streamed_text = ""
    saw_first_delta = False
    status_state: dict[str, object] = {
        "message": "첫 토큰 대기 중...",
        "started": time.perf_counter(),
        "done": False,
    }

    async def update_stream_status() -> None:
        while not status_state["done"]:
            ensure_not_stopped(stop_event)
            render_status_message(
                status_placeholder,
                str(status_state["message"]),
                time.perf_counter() - float(status_state["started"]),
            )
            await asyncio.sleep(0.25)

    status_task: asyncio.Task | None = None
    try:
        ensure_not_stopped(stop_event)
        append_run_event(mode_message)
        append_run_event("`Runner.run_streamed()` 시작")
        status_task = asyncio.create_task(update_stream_status())
        result = Runner.run_streamed(
            agent,
            prompt,
            session=session,
            max_turns=5,
        )

        async for event in result.stream_events():
            ensure_not_stopped(stop_event)
            if event.type == "run_item_stream_event":
                if event.name == "tool_called":
                    append_run_event("Agents SDK stream event: tool 호출 감지")
                    render_status_message(status_placeholder, "웹 검색 중...")
                elif event.name == "tool_output":
                    append_run_event("Agents SDK stream event: tool 결과 수신")
                    render_status_message(
                        status_placeholder,
                        "검색 결과를 바탕으로 답변 작성 중...",
                    )

            if event.type != "raw_response_event":
                continue

            data = getattr(event, "data", None)
            if getattr(data, "type", None) != "response.output_text.delta":
                continue

            delta = getattr(data, "delta", "")
            if not delta:
                continue

            if not saw_first_delta:
                saw_first_delta = True
                status_state["message"] = "답변 스트리밍 중..."
                status_state["started"] = time.perf_counter()
                append_run_event("응답 토큰 스트리밍 시작")
            streamed_text += delta
            response_placeholder.markdown(f"{streamed_text}▌")

        answer = linkify_plain_urls(str(result.final_output or streamed_text))
        response_placeholder.markdown(answer)
        status_state["done"] = True
        await status_task
        status_placeholder.empty()

        evidence = extract_run_evidence(result)
        evidence["total_seconds"] = time.perf_counter() - started
        evidence["streaming"] = True
        append_run_event("`Runner.run_streamed()` 완료")
        evidence["events"] = list(run_events)
        evidence = merge_search_timings(evidence, search_timings)
        return answer, evidence
    finally:
        status_state["done"] = True
        if status_task is not None and not status_task.done():
            await status_task
        SEARCH_CALL_COUNT.reset(search_count_token)
        RUN_EVENT_QUEUE.reset(queue_token)
        RUN_EVENT_RENDERER.reset(renderer_token)
        RUN_STARTED_AT.reset(started_token)
        RUN_EVENTS.reset(events_token)
        SEARCH_TIMINGS.reset(timing_token)


def run_agent_sync_timed(
    agent: Agent,
    prompt: str,
    session: SQLiteSession,
    activity_renderer: Callable[[list[dict[str, object]]], None] | None = None,
    event_queue: Queue | None = None,
    search_expected: bool = False,
) -> tuple[str, dict[str, object]]:
    started = time.perf_counter()
    search_timings: list[dict[str, object]] = []
    run_events: list[dict[str, object]] = []
    timing_token = SEARCH_TIMINGS.set(search_timings)
    events_token = RUN_EVENTS.set(run_events)
    started_token = RUN_STARTED_AT.set(started)
    renderer_token = RUN_EVENT_RENDERER.set(activity_renderer)
    queue_token = RUN_EVENT_QUEUE.set(event_queue)
    search_count_token = SEARCH_CALL_COUNT.set([0])

    try:
        append_run_event("`Runner.run_sync()` 시작")
        if search_expected:
            append_run_event("자동 모드: 검색/tool-call 질문이라 안정 실행")
        result = Runner.run_sync(
            agent,
            prompt,
            session=session,
            max_turns=5,
        )
        answer = linkify_plain_urls(str(result.final_output))
        evidence = extract_run_evidence(result)
        evidence["total_seconds"] = time.perf_counter() - started
        evidence["streaming"] = False
        append_run_event("`Runner.run_sync()` 완료")
        evidence["events"] = list(run_events)
        evidence = merge_search_timings(evidence, search_timings)
        return answer, evidence
    finally:
        SEARCH_CALL_COUNT.reset(search_count_token)
        RUN_EVENT_QUEUE.reset(queue_token)
        RUN_EVENT_RENDERER.reset(renderer_token)
        RUN_STARTED_AT.reset(started_token)
        RUN_EVENTS.reset(events_token)
        SEARCH_TIMINGS.reset(timing_token)


def prompt_likely_needs_search(prompt: str) -> bool:
    normalized = prompt.lower()
    return any(hint in normalized for hint in WEB_SEARCH_HINTS)


def initialize_state() -> None:
    if "chat_session_key" not in st.session_state:
        st.session_state.chat_session_key = get_query_session_key() or make_chat_session_key()
        set_query_session_key(st.session_state.chat_session_key)

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(st.session_state.chat_session_key)

    if "agent_session" not in st.session_state:
        st.session_state.agent_session = SQLiteSession(
            st.session_state.session_id,
            str(DB_PATH),
        )

    if "coaching_style" not in st.session_state:
        st.session_state.coaching_style = DEFAULT_COACHING_STYLE

    if "custom_coach_instructions" not in st.session_state:
        st.session_state.custom_coach_instructions = ""


def restore_messages_if_needed(force: bool = False) -> None:
    if "messages" in st.session_state and not force:
        return

    try:
        restored_messages = supabase_load_messages(st.session_state.chat_session_key)
    except PermissionError:
        restored_messages = []
        if current_auth_user_id():
            st.session_state.supabase_status = "Supabase 복원: 권한 확인 필요"
        else:
            st.session_state.supabase_status = "Google 로그인 후 대화 복원 가능"
    except Exception as exc:
        restored_messages = []
        st.session_state.supabase_status = f"Supabase 복원 실패: {exc.__class__.__name__}"

    if restored_messages:
        st.session_state.messages = restored_messages
        st.session_state.supabase_status = f"Supabase 복원: {len(restored_messages)}개 메시지"
    elif "messages" not in st.session_state:
        st.session_state.messages = [default_greeting()]


def reset_conversation() -> None:
    st.session_state.chat_session_key = make_chat_session_key()
    set_query_session_key(st.session_state.chat_session_key)
    st.session_state.session_id = str(st.session_state.chat_session_key)
    st.session_state.agent_session = SQLiteSession(
        st.session_state.session_id,
        str(DB_PATH),
    )
    st.session_state.messages = [new_conversation_greeting()]
    st.session_state.supabase_status = "새 Supabase 채팅 세션"
    user_id = current_auth_user_id()
    if user_id:
        try:
            supabase_attach_session_to_user(st.session_state.chat_session_key, user_id)
        except Exception as exc:
            st.session_state.supabase_status = (
                f"새 세션 사용자 연결 실패: {exc.__class__.__name__}"
            )


def reset_agent_session_only() -> None:
    st.session_state.session_id = f"life-coach-{uuid.uuid4().hex}"
    st.session_state.agent_session = SQLiteSession(
        st.session_state.session_id,
        str(DB_PATH),
    )


def run_sync_with_bad_request_recovery(
    agent: Agent,
    prompt: str,
    activity_renderer: Callable[[list[dict[str, object]]], None] | None = None,
    event_queue: Queue | None = None,
    search_expected: bool = False,
) -> tuple[str, dict[str, object]]:
    try:
        return run_agent_sync_timed(
            agent,
            prompt,
            st.session_state.agent_session,
            activity_renderer=activity_renderer,
            event_queue=event_queue,
            search_expected=search_expected,
        )
    except Exception as exc:
        if exc.__class__.__name__ != "BadRequestError":
            raise

        reset_agent_session_only()
        answer, evidence = run_agent_sync_timed(
            agent,
            prompt,
            st.session_state.agent_session,
            activity_renderer=activity_renderer,
            event_queue=event_queue,
            search_expected=search_expected,
        )
        evidence["session_recovered"] = True
        return answer, evidence


def run_agent_sync_timed_with_recovery(
    agent: Agent,
    prompt: str,
    session: SQLiteSession,
    event_queue: Queue | None,
    search_expected: bool,
) -> tuple[str, dict[str, object], str | None, SQLiteSession | None]:
    try:
        answer, evidence = run_agent_sync_timed(
            agent,
            prompt,
            session,
            event_queue=event_queue,
            search_expected=search_expected,
        )
        return answer, evidence, None, None
    except Exception as exc:
        if exc.__class__.__name__ != "BadRequestError":
            raise

        session_id = f"life-coach-{uuid.uuid4().hex}"
        recovered_session = SQLiteSession(session_id, str(DB_PATH))
        answer, evidence = run_agent_sync_timed(
            agent,
            prompt,
            recovered_session,
            event_queue=event_queue,
            search_expected=search_expected,
        )
        evidence["session_recovered"] = True
        return answer, evidence, session_id, recovered_session


def run_search_for_streaming_answer(
    agent: Agent,
    prompt: str,
    event_queue: Queue | None,
    started_at: float,
) -> tuple[str, dict[str, object], list[dict[str, object]]]:
    search_timings: list[dict[str, object]] = []
    goal_timings: list[dict[str, object]] = []
    run_events: list[dict[str, object]] = []
    timing_token = SEARCH_TIMINGS.set(search_timings)
    goals_token = GOALS_TIMINGS.set(goal_timings)
    events_token = RUN_EVENTS.set(run_events)
    started_token = RUN_STARTED_AT.set(started_at)
    renderer_token = RUN_EVENT_RENDERER.set(None)
    queue_token = RUN_EVENT_QUEUE.set(event_queue)
    search_count_token = SEARCH_CALL_COUNT.set([0])
    goal_count_token = GOAL_SEARCH_CALL_COUNT.set([0])

    try:
        append_run_event("자동 모드: 개인 목표/웹 검색 먼저 안정 실행")
        append_run_event("`Runner.run_sync()` 검색 단계 시작")
        search_session = SQLiteSession(
            f"life-coach-search-{uuid.uuid4().hex}",
            str(DB_PATH),
        )
        result = Runner.run_sync(
            agent,
            prompt,
            session=search_session,
            max_turns=4,
        )
        evidence = extract_run_evidence(result)
        append_run_event("`Runner.run_sync()` 검색 단계 완료")
        evidence["events"] = list(run_events)
        evidence = merge_search_timings(evidence, search_timings)
        if goal_timings:
            evidence["goal_searches"] = [
                {
                    "query": timing.get("query"),
                    "seconds": timing.get("seconds"),
                }
                for timing in goal_timings
            ]

        context_sections: list[str] = []
        goal_parts = [
            str(timing.get("output", ""))
            for timing in goal_timings
            if timing.get("output")
        ]
        if goal_parts:
            context_sections.append("[개인 목표/기록]\n" + "\n\n".join(goal_parts))
        web_parts = [
            str(timing.get("output", ""))
            for timing in search_timings
            if timing.get("output")
        ]
        if web_parts:
            context_sections.append("[웹 검색 결과]\n" + "\n\n".join(web_parts))
        return "\n\n".join(context_sections), evidence, list(run_events)
    finally:
        GOAL_SEARCH_CALL_COUNT.reset(goal_count_token)
        GOALS_TIMINGS.reset(goals_token)
        SEARCH_CALL_COUNT.reset(search_count_token)
        RUN_EVENT_QUEUE.reset(queue_token)
        RUN_EVENT_RENDERER.reset(renderer_token)
        RUN_STARTED_AT.reset(started_token)
        RUN_EVENTS.reset(events_token)
        SEARCH_TIMINGS.reset(timing_token)


def render_auth_controls() -> None:
    user = current_auth_user()
    auth_status = st.session_state.get("auth_status")
    if auth_status and "bad_oauth_state" in str(auth_status):
        auth_status = "Google 로그인 링크를 갱신했어요. 다시 로그인해 주세요."
        st.session_state.auth_status = auth_status

    if user:
        display_name = user.get("email") or user.get("name") or "Google user"
        st.caption(f"Google 로그인: {display_name}")
        if st.button("로그아웃", use_container_width=True):
            refresh_token = st.session_state.get("auth_refresh_token")
            if refresh_token:
                supabase_revoke_refresh_token(str(refresh_token))
            cookie_token = st.session_state.get("auth_cookie_token") or get_auth_cookie_token()
            if cookie_token:
                supabase_revoke_app_auth_session(str(cookie_token))
            for key in (
                "auth_user",
                "auth_access_token",
                "auth_refresh_token",
                "auth_cookie_token",
                "preferences_loaded_for_user",
                "preference_status",
            ):
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.coaching_style = DEFAULT_COACHING_STYLE
            st.session_state.custom_coach_instructions = ""
            clear_cached_google_oauth_url()
            st.session_state.pending_auth_cookie_delete = True
            st.session_state.auth_status = "Google 로그아웃 완료"
            st.rerun()
        render_user_session_list()
    else:
        try:
            login_url = build_google_oauth_url()
        except Exception as exc:
            login_url = None
            st.caption(f"Google 로그인 준비 실패: {exc.__class__.__name__}")
        if login_url:
            safe_url = html.escape(login_url, quote=True)
            st.markdown(
                f"""
<a class="google-login-link" href="{safe_url}" target="_self" rel="noreferrer">
  Google로 로그인
</a>
""",
                unsafe_allow_html=True,
            )
        else:
            st.caption("Google 로그인: Supabase 설정 필요")

    if auth_status:
        st.caption(str(auth_status))


def render_share_controls(session_key: str, user_id: str) -> None:
    with st.expander("공유", expanded=False):
        st.caption("공유 링크는 현재 대화의 읽기 전용 snapshot입니다.")
        messages = st.session_state.get("messages") or []
        if not isinstance(messages, list):
            messages = []

        can_share = any(
            isinstance(message, dict) and message.get("role") == "user"
            for message in messages
        )
        if not can_share:
            st.caption("사용자 메시지가 생기면 공유 링크를 만들 수 있습니다.")
            return

        if st.button(
            "공유하기",
            key=f"create-share-{session_key[-8:]}",
            use_container_width=True,
        ):
            try:
                share = supabase_create_shared_chat(session_key, user_id, messages)
                share_url = share["url"]
                st.session_state.share_status = (
                    "공유 링크를 만들었어요. 아래 '공유 앱 선택' 또는 "
                    "'링크 복사'를 눌러 공유하세요."
                )
                st.session_state.latest_share_url = share_url
            except Exception as exc:
                st.session_state.share_status = (
                    f"공유 링크 생성 실패: {exc.__class__.__name__}"
                )

        share_status = st.session_state.get("share_status")
        if share_status:
            st.caption(str(share_status))

        latest_share_url = st.session_state.get("latest_share_url")
        latest_share_url_text = ""
        if latest_share_url:
            latest_share_url = str(latest_share_url)
            latest_share_url_text = latest_share_url
            st.text_input(
                "최근 공유 링크",
                value=latest_share_url,
                key=f"latest-share-url-{session_key[-8:]}",
            )
            render_web_share_actions(
                latest_share_url,
                f"latest-share-{session_key[-8:]}",
            )

        try:
            shares = supabase_list_shared_chats(session_key, user_id)
        except Exception as exc:
            st.caption(f"공유 링크 로드 실패: {exc.__class__.__name__}")
            return

        if not shares:
            return

        st.caption("활성 공유 링크")
        for item in shares:
            share_token = str(item.get("share_token") or "")
            if not share_token:
                continue
            share_url = build_share_url(share_token)
            created_at = str(item.get("created_at") or "")[:10]
            st.text_input(
                f"공유 링크 {created_at}",
                value=share_url,
                key=f"share-url-{share_token[-8:]}",
            )
            if share_url != latest_share_url_text:
                render_web_share_actions(
                    share_url,
                    f"share-{share_token[-8:]}",
                )
            if st.button(
                "공유 취소",
                key=f"revoke-share-{share_token[-8:]}",
                use_container_width=True,
            ):
                try:
                    supabase_revoke_shared_chat(share_token, user_id)
                    st.session_state.share_status = "공유 링크를 취소했어요."
                    if st.session_state.get("latest_share_url") == share_url:
                        del st.session_state.latest_share_url
                except Exception as exc:
                    st.session_state.share_status = (
                        f"공유 취소 실패: {exc.__class__.__name__}"
                    )
                st.rerun()


def render_user_session_list() -> None:
    user_id = current_auth_user_id()
    if not user_id:
        return

    with st.expander("내 대화", expanded=True):
        try:
            sessions = supabase_list_user_sessions(user_id)
        except Exception as exc:
            st.caption(f"대화 목록 로드 실패: {exc.__class__.__name__}")
            return

        if not sessions:
            st.caption("아직 저장된 대화가 없습니다.")
            return

        current_session_key = str(st.session_state.get("chat_session_key") or "")
        for index, item in enumerate(sessions):
            session_key = str(item.get("session_key") or "")
            if not session_key:
                continue
            label = format_saved_session_label(item)
            is_current = session_key == current_session_key

            if is_current:
                st.caption(f"현재 대화: {label}")
                title_value = str(item.get("title") or "").strip()
                updated_title = st.text_input(
                    "대화 이름",
                    value=title_value,
                    placeholder="대화 이름",
                    key=f"session-title-{session_key[-8:]}",
                    label_visibility="collapsed",
                )
                rename_col, delete_col = st.columns(2, gap="small")
                confirm_key = f"confirm-delete-{session_key}"
                with rename_col:
                    if st.session_state.get(confirm_key):
                        if st.button(
                            "삭제 취소",
                            key=f"cancel-delete-session-{session_key[-8:]}",
                            use_container_width=True,
                        ):
                            del st.session_state[confirm_key]
                            st.rerun()
                    else:
                        if st.button(
                            "이름 저장",
                            key=f"rename-session-{session_key[-8:]}",
                            use_container_width=True,
                        ):
                            try:
                                supabase_update_session_title(
                                    session_key,
                                    user_id,
                                    updated_title,
                                )
                                st.session_state.supabase_status = "대화 이름 저장됨"
                            except Exception as exc:
                                st.session_state.supabase_status = (
                                    f"대화 이름 저장 실패: {exc.__class__.__name__}"
                                )
                            st.rerun()
                with delete_col:
                    if st.session_state.get(confirm_key):
                        if st.button(
                            "삭제 확정",
                            key=f"delete-session-confirm-{session_key[-8:]}",
                            use_container_width=True,
                        ):
                            deleted = False
                            try:
                                supabase_delete_session(session_key, user_id)
                                st.session_state.supabase_status = "대화 삭제됨"
                                deleted = True
                            except Exception as exc:
                                st.session_state.supabase_status = (
                                    f"대화 삭제 실패: {exc.__class__.__name__}"
                                )
                            if confirm_key in st.session_state:
                                del st.session_state[confirm_key]
                            if deleted:
                                reset_conversation()
                            st.rerun()
                    elif st.button(
                        "삭제",
                        key=f"delete-session-{session_key[-8:]}",
                        use_container_width=True,
                    ):
                        st.session_state[confirm_key] = True
                        st.rerun()
                render_share_controls(session_key, user_id)
                continue

            if st.button(
                label,
                key=f"saved-session-{index}-{session_key[-8:]}",
                use_container_width=True,
            ):
                switch_conversation(session_key)
                st.rerun()


def render_shared_chat_page(share_token: str) -> None:
    st.title(APP_TITLE)
    st.caption("읽기 전용 공유 대화")

    try:
        shared_chat = supabase_load_shared_chat(share_token)
    except Exception as exc:
        st.error(f"공유 대화를 불러오지 못했어요. 오류 유형: {exc.__class__.__name__}")
        st.markdown(f"[새 대화 시작하기](<{read_app_base_url()}>)")
        return

    if not shared_chat:
        st.warning("공유 링크가 없거나 취소되었어요.")
        st.markdown(f"[새 대화 시작하기](<{read_app_base_url()}>)")
        return

    title = str(shared_chat.get("title") or "공유된 대화").strip()
    shared_at = format_shared_timestamp(shared_chat.get("created_at"))
    if title:
        st.subheader(title)
    caption_parts = ["공유 시점의 snapshot입니다."]
    if shared_at:
        caption_parts.append(f"공유 시각: {shared_at}")
    st.info(" ".join(caption_parts))

    raw_messages = shared_chat.get("messages")
    messages = raw_messages if isinstance(raw_messages, list) else []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        with st.chat_message(role):
            st.markdown(linkify_plain_urls(content))
            copy_label = "프롬프트 복사" if role == "user" else "출력 복사"
            render_copy_button(
                content,
                f"copy-share-{share_token[-8:]}-{index}-{role}",
                copy_label,
            )

    st.divider()
    st.markdown(f"[내 Life Coach 대화 시작하기](<{read_app_base_url()}>)")


def load_default_goals_text() -> str:
    try:
        return GOALS_PATH.read_text(encoding="utf-8")[:GOALS_MAX_CHARS]
    except OSError:
        return ""


def extract_uploaded_goal_text(uploaded_file) -> str:
    name = (getattr(uploaded_file, "name", "") or "").lower()
    # Streamlit UploadedFile exposes getvalue() and stays re-readable across
    # reruns; fall back to read() only for plain file-like objects.
    if hasattr(uploaded_file, "getvalue"):
        data = uploaded_file.getvalue()
    else:
        data = uploaded_file.read()
    if name.endswith(".pdf"):
        try:
            import io

            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(data))
            parts: list[str] = []
            total = 0
            for page in reader.pages:
                text = page.extract_text() or ""
                parts.append(text)
                total += len(text)
                if total >= GOALS_MAX_CHARS:
                    break
            return "\n\n".join(parts)[:GOALS_MAX_CHARS]
        except Exception:
            return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")[:GOALS_MAX_CHARS]
    return str(data)[:GOALS_MAX_CHARS]


def render_goals_panel() -> None:
    with st.expander("개인 목표 파일", expanded=False):
        st.caption(
            "목표·일기 파일(TXT/Markdown/PDF)을 올리거나 샘플을 불러오면 코치가 "
            "`search_goals` 도구로 내용을 참고해 답합니다. 로드하지 않으면 일반 "
            "대화로 동작합니다."
        )
        nonce = int(st.session_state.get("goals_uploader_nonce", 0))
        uploaded = st.file_uploader(
            "목표 파일 업로드",
            type=["txt", "md", "markdown", "pdf"],
            key=f"goals-file-uploader-{nonce}",
            label_visibility="collapsed",
        )
        if uploaded is not None:
            extracted = extract_uploaded_goal_text(uploaded)
            if extracted.strip():
                st.session_state.goals_text = extracted
                st.session_state.goals_source = f"업로드: {uploaded.name}"
            else:
                st.caption(
                    "파일에서 텍스트를 추출하지 못했어요. PDF라면 텍스트형 PDF인지 "
                    "확인하거나 TXT/Markdown으로 올려 주세요."
                )

        goals_text = str(st.session_state.get("goals_text") or "")
        source = str(st.session_state.get("goals_source") or "")
        if goals_text.strip():
            st.caption(f"현재 목표 문서: {source} · {len(goals_text)}자")
            preview = goals_text[:GOALS_PREVIEW_CHARS]
            if len(goals_text) > GOALS_PREVIEW_CHARS:
                preview += " ..."
            st.text_area("목표 미리보기", value=preview, height=140, disabled=True)
            if st.button(
                "목표 문서 비우기", use_container_width=True, key="clear-goals"
            ):
                for key in ("goals_text", "goals_source"):
                    if key in st.session_state:
                        del st.session_state[key]
                # rotate the uploader key so the previously uploaded file is
                # dropped and does not silently reload after clearing.
                st.session_state.goals_uploader_nonce = nonce + 1
                st.rerun()
        else:
            st.caption("로드된 목표 문서가 없습니다. (일반 대화 모드)")
            if st.button(
                "샘플 목표 불러오기",
                use_container_width=True,
                key="load-sample-goals",
            ):
                default_text = load_default_goals_text()
                if default_text.strip():
                    st.session_state.goals_text = default_text
                    st.session_state.goals_source = (
                        "기본 샘플 (goals/personal_goals.md)"
                    )
                    st.rerun()
                else:
                    st.caption("샘플 목표 파일을 찾지 못했어요.")


def render_coaching_preferences() -> None:
    user_id = current_auth_user_id()
    with st.expander("코칭 스타일", expanded=False):
        style_options = list(COACHING_STYLES.keys())
        current_style = normalize_coaching_style(st.session_state.get("coaching_style"))
        if current_style not in style_options:
            current_style = DEFAULT_COACHING_STYLE
        if st.session_state.get("coaching_style") != current_style:
            st.session_state.coaching_style = current_style

        selected_style = st.segmented_control(
            "답변 톤",
            options=style_options,
            required=True,
            format_func=coaching_style_label,
            key="coaching_style",
            width="stretch",
        )
        normalized_style = normalize_coaching_style(str(selected_style))
        st.caption(coaching_style_description(normalized_style))

        st.text_area(
            "직접 코칭 지시",
            placeholder=(
                "예: 너무 길게 말하지 말고, 마지막에 오늘 할 일 1개만 물어봐줘."
            ),
            max_chars=CUSTOM_INSTRUCTIONS_MAX_CHARS,
            key="custom_coach_instructions",
            height=96,
        )

        preference_status = st.session_state.get("preference_status")
        if preference_status:
            st.caption(str(preference_status))

        if st.button("코칭 설정 저장", use_container_width=True):
            style = normalize_coaching_style(st.session_state.get("coaching_style"))
            custom = clean_custom_instructions(
                st.session_state.get("custom_coach_instructions")
            )
            if user_id:
                try:
                    supabase_upsert_user_preferences(user_id, style, custom)
                    st.session_state.preferences_loaded_for_user = user_id
                    st.session_state.preference_status = "코칭 설정 저장됨"
                except Exception as exc:
                    st.session_state.preference_status = (
                        f"코칭 설정 저장 실패: {exc.__class__.__name__}"
                    )
            else:
                st.session_state.preference_status = (
                    "현재 브라우저 세션에만 적용돼요. 로그인하면 저장할 수 있습니다."
                )
            st.rerun()


def render_sidebar() -> None:
    with st.sidebar:
        st.header("설정")

        if st.button("새 대화", use_container_width=True):
            reset_conversation()
            st.rerun()

        render_auth_controls()
        render_coaching_preferences()
        render_goals_panel()

        sidebar_status = str(st.session_state.get("supabase_status") or "")
        if any(marker in sidebar_status for marker in ("실패", "필요", "권한", "미설정")):
            st.caption(sidebar_status)


def render_prompt_settings() -> tuple[str, str]:
    configured_model = os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL)
    if configured_model not in SUPPORTED_MODELS:
        configured_model = DEFAULT_MODEL

    configured_thinking = normalize_thinking_mode(os.getenv("DEEPSEEK_THINKING_MODE"))
    model_column, thinking_column = st.columns([1.05, 1], gap="small")

    with model_column:
        model = st.segmented_control(
            "모델",
            options=SUPPORTED_MODELS,
            default=configured_model,
            required=True,
            format_func=model_label,
            key="model-select",
            width="stretch",
        )

    thinking_options = list(THINKING_MODES.keys())
    with thinking_column:
        thinking_mode = st.segmented_control(
            "사고 모드",
            options=thinking_options,
            default=configured_thinking,
            required=True,
            format_func=thinking_mode_label,
            key="thinking-mode-select",
            width="stretch",
        )

    return str(model or DEFAULT_MODEL), normalize_thinking_mode(str(thinking_mode))


def drain_event_queue(
    event_queue: Queue,
    latest_events: list[dict[str, object]],
) -> list[dict[str, object]]:
    while True:
        try:
            latest_events = event_queue.get_nowait()
        except Empty:
            return latest_events


def active_message_for_events(events: list[dict[str, object]]) -> str:
    if not events:
        return "Runner 시작 대기 중..."

    message = str(events[-1].get("message", ""))
    if "tool 호출" in message and "완료" not in message:
        return "웹 검색 중..."
    if "모델 답변 생성 대기 시작" in message or "tool 완료" in message:
        return "모델 답변 생성 중..."
    if "응답 토큰 스트리밍 시작" in message:
        return "스트리밍 중..."
    if "Runner.run_sync() 완료" in message or "Runner.run_streamed() 완료" in message:
        return ""
    return "다음 단계 대기 중..."


def run_sync_with_live_activity(
    agent: Agent,
    prompt: str,
    activity_placeholder,
    status_placeholder,
    search_expected: bool,
) -> tuple[str, dict[str, object]]:
    event_queue: Queue = Queue()
    session = st.session_state.agent_session
    activity_started = time.perf_counter()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            run_agent_sync_timed_with_recovery,
            agent,
            prompt,
            session,
            event_queue=event_queue,
            search_expected=search_expected,
        )

        latest_events: list[dict[str, object]] = []
        while not future.done():
            latest_events = drain_event_queue(event_queue, latest_events)
            if latest_events:
                last_seconds = float(latest_events[-1].get("seconds") or 0)
                active_message = active_message_for_events(latest_events)
                active_seconds = max(
                    0.0,
                    time.perf_counter() - activity_started - last_seconds,
                )
                activity_placeholder.markdown(format_run_events_markdown(latest_events))
                if active_message:
                    render_status_message(
                        status_placeholder,
                        active_message,
                        active_seconds,
                    )
            time.sleep(0.1)

        latest_events = drain_event_queue(event_queue, latest_events)
        if latest_events:
            activity_placeholder.markdown(format_run_events_markdown(latest_events))

        answer, evidence, recovered_session_id, recovered_session = future.result()
        if recovered_session_id and recovered_session:
            st.session_state.session_id = recovered_session_id
            st.session_state.agent_session = recovered_session

        return answer, evidence


def run_search_with_live_activity(
    agent: Agent,
    prompt: str,
    activity_placeholder,
    status_placeholder,
) -> tuple[str, dict[str, object], list[dict[str, object]], float]:
    event_queue: Queue = Queue()
    started = time.perf_counter()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            run_search_for_streaming_answer,
            agent,
            prompt,
            event_queue,
            started,
        )

        latest_events: list[dict[str, object]] = []
        while not future.done():
            latest_events = drain_event_queue(event_queue, latest_events)
            if latest_events:
                last_seconds = float(latest_events[-1].get("seconds") or 0)
                active_message = active_message_for_events(latest_events)
                active_seconds = max(0.0, time.perf_counter() - started - last_seconds)
                activity_placeholder.markdown(format_run_events_markdown(latest_events))
                if active_message:
                    render_status_message(
                        status_placeholder,
                        active_message,
                        active_seconds,
                    )
            time.sleep(0.1)

        latest_events = drain_event_queue(event_queue, latest_events)
        if latest_events:
            activity_placeholder.markdown(format_run_events_markdown(latest_events))

        search_context, search_evidence, search_events = future.result()
        return search_context, search_evidence, search_events, started


async def run_search_then_stream_answer(
    search_agent: Agent,
    answer_agent: Agent,
    prompt: str,
    session: SQLiteSession,
    activity_placeholder,
    response_placeholder,
    status_placeholder,
    stop_event: threading.Event | None = None,
) -> tuple[str, dict[str, object]]:
    ensure_not_stopped(stop_event)
    search_context, search_evidence, search_events, started = run_search_with_live_activity(
        search_agent,
        prompt,
        activity_placeholder,
        status_placeholder,
    )
    ensure_not_stopped(stop_event)
    render_status_message(status_placeholder, "답변 스트리밍 중...")

    augmented_prompt = (
        f"{prompt}\n\n"
        "[참고 자료]\n"
        f"{search_context or '참고할 개인 목표/검색 결과를 가져오지 못했습니다.'}\n\n"
        "위 개인 목표/기록과 검색 결과를 바탕으로, 사용자의 목표와 최근 진행 상황을 "
        "반영해 바로 실행 가능한 라이프 코칭 답변을 해 주세요."
    )

    def render_activity(events: list[dict[str, object]]) -> None:
        activity_placeholder.markdown(format_run_events_markdown(events))

    answer, stream_evidence = await run_agent_streamed(
        answer_agent,
        augmented_prompt,
        session,
        response_placeholder,
        status_placeholder,
        activity_renderer=render_activity,
        initial_events=search_events,
        started_at=started,
        mode_message="검색 결과 기반 최종 답변 스트리밍",
        stop_event=stop_event,
    )

    stream_evidence["searches"] = search_evidence.get("searches", [])
    stream_evidence["events"] = stream_evidence.get("events") or search_events
    stream_evidence["total_seconds"] = time.perf_counter() - started
    stream_evidence["search_then_stream"] = True
    return answer, stream_evidence


def main() -> None:
    st.set_page_config(page_title=APP_SHORT_TITLE, page_icon=str(APP_ICON_PATH))
    st.markdown(
        """
<style>
footer,
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
.stDeployButton,
[data-testid="stFooter"],
[class*="viewerBadge"],
[class*="ViewerBadge"],
a[href*="github.com/twsftrp-arch/life-coach-agent"],
a[href*="streamlit.io/cloud"],
a[href*="streamlit.io/"] {
  display: none !important;
  visibility: hidden !important;
}
[data-testid="stAppViewContainer"] .main .block-container,
[data-testid="stMainBlockContainer"] {
  padding-bottom: 13rem;
}
.run-status-box {
  align-items: center;
  background: rgba(46, 134, 222, 0.08);
  border: 1px solid rgba(46, 134, 222, 0.22);
  border-radius: 6px;
  display: flex;
  gap: 0.55rem;
  left: 50%;
  max-width: 46rem;
  min-height: 42px;
  padding: 0.55rem 0.75rem;
  position: fixed;
  bottom: 12.85rem;
  right: auto;
  transform: translateX(-50%);
  width: calc(100% - 2rem);
  z-index: 10020;
}
.run-status-box code {
  font-variant-numeric: tabular-nums;
  margin-left: auto;
  min-width: 4ch;
  text-align: right;
}
.run-status-dot {
  animation: run-status-pulse 1s ease-in-out infinite;
  background: #2e86de;
  border-radius: 999px;
  display: inline-block;
  flex: 0 0 auto;
  height: 0.58rem;
  width: 0.58rem;
}
.run-status-text {
  line-height: 1.25;
}
.floating-stop-anchor {
  display: none;
}
.google-login-link {
  align-items: center;
  border: 1px solid rgba(49, 51, 63, 0.18);
  border-radius: 6px;
  color: inherit;
  display: flex;
  font-size: 0.92rem;
  font-weight: 600;
  justify-content: center;
  margin: 0.35rem 0 0.75rem;
  min-height: 38px;
  text-decoration: none;
}
.google-login-link:hover {
  border-color: rgba(46, 134, 222, 0.55);
  color: #1d4ed8;
  text-decoration: none;
}
div[class*="st-key-model-select"],
div[class*="st-key-thinking-mode-select"] {
  background: rgba(255, 255, 255, 0.96);
  border: 1px solid rgba(49, 51, 63, 0.16);
  border-radius: 6px;
  box-shadow: 0 10px 28px rgba(15, 23, 42, 0.12);
  left: 50%;
  max-width: 46rem;
  padding: 0.28rem 0.36rem;
  position: fixed;
  right: auto;
  transform: translateX(-50%);
  width: calc(100% - 2rem) !important;
  z-index: 9980;
}
div[class*="st-key-model-select"] {
  bottom: 9.95rem;
}
div[class*="st-key-thinking-mode-select"] {
  bottom: 7.35rem;
}
div[class*="st-key-model-select"] label,
div[class*="st-key-thinking-mode-select"] label {
  display: none;
}
div[class*="st-key-model-select"] button,
div[class*="st-key-thinking-mode-select"] button {
  font-size: 0.78rem;
  min-height: 30px;
  padding: 0.25rem 0.45rem;
  white-space: nowrap;
}
div[class*="st-key-stop-run-"] {
  bottom: 4.1rem;
  position: fixed;
  right: 5.25rem;
  width: 4.1rem;
  z-index: 10030;
}
div[class*="st-key-stop-run-"] button {
  background: #ffffff;
  border: 1px solid rgba(190, 18, 60, 0.42);
  border-radius: 6px;
  box-shadow: 0 8px 24px rgba(15, 23, 42, 0.16);
  color: #be123c;
  min-height: 38px;
  padding: 0.38rem 0.7rem;
}
div[class*="st-key-stop-run-"] button:hover {
  border-color: rgba(190, 18, 60, 0.7);
  color: #9f1239;
}
@media (min-width: 900px) {
  body:has([data-testid="stSidebar"][aria-expanded="true"]) .run-status-box,
  body:has([data-testid="stSidebar"][aria-expanded="true"]) div[class*="st-key-model-select"],
  body:has([data-testid="stSidebar"][aria-expanded="true"]) div[class*="st-key-thinking-mode-select"] {
    left: calc(50% + 150px);
  }
  .run-status-box {
    width: 46rem;
  }
  div[class*="st-key-model-select"],
  div[class*="st-key-thinking-mode-select"] {
    width: 46rem !important;
  }
  div[class*="st-key-stop-run-"] {
    right: calc((100vw - 46rem) / 2 + 5.25rem);
  }
}
@keyframes run-status-pulse {
  0%, 100% { opacity: 0.35; transform: scale(0.82); }
  50% { opacity: 1; transform: scale(1); }
}
</style>
""",
        unsafe_allow_html=True,
    )
    render_browser_head_tags()

    share_token = get_query_share_token()
    if share_token:
        render_copy_feedback()
        render_shared_chat_page(share_token)
        return

    initialize_state()
    handle_google_oauth_callback()
    restore_auth_session_if_possible()
    restore_user_preferences_if_possible()
    stale_permission_status = (
        st.session_state.get("supabase_status") == "Supabase 복원 실패: PermissionError"
    )
    restore_messages_if_needed(
        force=bool(current_auth_user_id() and stale_permission_status)
    )
    render_auth_cookie_scripts()

    render_copy_feedback()

    render_sidebar()
    api_key = read_deepseek_api_key()

    st.title(APP_TITLE)
    st.caption("OpenAI Agents SDK + Streamlit 기반 자기계발 코치")

    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(linkify_plain_urls(message["content"]))
            copy_label = (
                "프롬프트 복사" if message["role"] == "user" else "출력 복사"
            )
            render_copy_button(
                message["content"],
                f"copy-history-{index}-{message['role']}",
                copy_label,
            )
            render_run_evidence(message.get("evidence"))

    model, thinking_mode = render_prompt_settings()
    prompt = st.chat_input("예: 아침에 일찍 일어나고 싶은데 자꾸 알람을 꺼요")
    if not prompt:
        return

    run_id = f"run-{uuid.uuid4().hex}"
    st.session_state.messages.append({"role": "user", "content": prompt})
    persist_chat_message("user", prompt)
    with st.chat_message("user"):
        st.markdown(linkify_plain_urls(prompt))
        render_copy_button(prompt, f"copy-user-{run_id}", "프롬프트 복사")

    if not api_key:
        answer = (
            "모델 API 키가 필요합니다. Streamlit Secrets 또는 로컬 환경변수에 키를 넣어 주세요."
        )
        st.session_state.messages.append({"role": "assistant", "content": answer})
        persist_chat_message("assistant", answer)
        with st.chat_message("assistant"):
            st.warning(answer)
        return

    goals_text = str(st.session_state.get("goals_text") or "")
    # When a personal goal document is loaded, take the research path so the
    # coach searches the goals (and the web) before answering.
    needs_search = prompt_likely_needs_search(prompt) or bool(goals_text.strip())
    use_streaming = not needs_search
    stop_event = threading.Event()
    STOP_EVENTS[run_id] = stop_event
    st.session_state.stop_requested = False

    stop_placeholder = st.empty()
    with stop_placeholder.container():
        st.markdown(
            '<span class="floating-stop-anchor" aria-hidden="true"></span>',
            unsafe_allow_html=True,
        )
        st.button(
            "중지",
            key=f"stop-{run_id}",
            on_click=request_stop,
            args=(run_id,),
        )

    with st.chat_message("assistant"):
        activity_placeholder = st.empty()
        response_placeholder = st.empty()
        status_placeholder = st.empty()
        copy_placeholder = st.empty()

        def render_activity(events: list[dict[str, object]]) -> None:
            activity_placeholder.markdown(format_run_events_markdown(events))

        try:
            if use_streaming and not needs_search:
                agent = build_streaming_coach_agent(model, api_key, thinking_mode)
                try:
                    answer, evidence = asyncio.run(
                        run_agent_streamed(
                            agent,
                            prompt,
                            st.session_state.agent_session,
                            response_placeholder,
                            status_placeholder,
                            activity_renderer=render_activity,
                            stop_event=stop_event,
                        )
                    )
                except Exception as stream_exc:
                    status_placeholder.warning(
                        "스트리밍이 실패해 새 SDK 세션에서 동기 실행으로 재시도합니다."
                    )
                    reset_agent_session_only()
                    answer, evidence = run_agent_sync_timed(
                        agent,
                        prompt,
                        st.session_state.agent_session,
                        activity_renderer=render_activity,
                        search_expected=needs_search,
                    )
                    evidence["fallback_reason"] = stream_exc.__class__.__name__
                    evidence["session_recovered"] = True
                    response_placeholder.markdown(answer)
                    status_placeholder.empty()
            else:
                search_agent = build_search_agent(
                    model, api_key, thinking_mode, goals_text=goals_text
                )
                answer_agent = build_streaming_coach_agent(model, api_key, thinking_mode)
                answer, evidence = asyncio.run(
                    run_search_then_stream_answer(
                        search_agent,
                        answer_agent,
                        prompt,
                        st.session_state.agent_session,
                        activity_placeholder,
                        response_placeholder,
                        status_placeholder,
                        stop_event=stop_event,
                    )
                )
                status_placeholder.empty()
        except GenerationStopped:
            status_placeholder.empty()
            answer = "응답 생성을 중지했어요."
            evidence = {
                "model": model,
                "searches": [],
                "total_seconds": None,
                "stopped": True,
            }
            response_placeholder.warning(answer)
        except Exception as exc:
            status_placeholder.empty()
            answer = (
                "응답 생성 중 오류가 발생했어요. API 키, 모델 이름, 네트워크 상태를 확인해 주세요. "
                f"오류 유형: `{exc.__class__.__name__}`"
            )
            evidence = {"model": model, "searches": [], "total_seconds": None}
            response_placeholder.warning(answer)

        evidence = attach_runtime_settings(evidence, model, thinking_mode)
        stop_placeholder.empty()
        STOP_EVENTS.pop(run_id, None)
        activity_placeholder.empty()
        if answer and not evidence.get("stopped"):
            with copy_placeholder:
                render_copy_button(answer, f"copy-{run_id}", "출력 복사")
        render_run_evidence(evidence)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "evidence": evidence}
    )
    persist_chat_message("assistant", answer, evidence)


if __name__ == "__main__":
    main()
