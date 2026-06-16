# Life Coach Agent

Streamlit 채팅 UI와 OpenAI Agents SDK를 사용한 라이프 코치 에이전트입니다.
모델은 OpenAI-compatible Chat Completions API를 사용합니다.

## 기능

- `st.chat_input`, `st.chat_message` 기반 채팅 인터페이스
- 프롬프트 입력창에 붙어 보이는 모델/사고 모드 선택 `segmented_control`
- OpenAI Agents SDK의 `Agent`, `Runner` 사용
- `SQLiteSession` 기반 세션 메모리
- Supabase 기반 채팅 히스토리 저장/복원
- Google OAuth 로그인
- 로그인 사용자 기준 사이드바 대화 목록, 이름 변경, 삭제
- 읽기 전용 공유 링크 생성/취소
- 사용자별 코칭 스타일 preset과 직접 코칭 지시 저장
- 로그인 유지용 앱 세션 쿠키
- `function_tool` 기반 웹 검색
- 검색 질문의 검색 후 답변 스트리밍
- 검색 없는 대화의 자동 스트리밍
- OpenAI-compatible provider의 tool-call 호환성 보호를 위한 안정 실행 fallback
- 답변 전 `실시간 실행 로그`로 Runner 시작, tool 호출, tool 완료, 모델 답변 생성 상태를 실시간 표시
- 답변 완료 후 `실행 확인` 한 줄 요약과 `상세 실행 정보` 접힘 영역 표시
- 스트리밍 중 상태 박스 0.1초 단위 초시계 표시
- 응답 생성 중 입력창 전송 버튼 왼쪽에 `중지` 버튼 표시
- 사용자 프롬프트와 코치 답변의 복사 버튼 표시
- 총 응답 시간과 웹 검색 도구 실행 시간 표시
- 동기부여, 자기 개발, 습관 형성 조언에 맞춘 한국어 코칭 응답

## 구조

- 모델 연결: `AsyncOpenAI(api_key=..., base_url=...)`
- Agents SDK 모델: `OpenAIChatCompletionsModel`
- 모델 선택: 내부 모델 id는 `deepseek-v4-flash`, `deepseek-v4-pro`를 쓰고, 사용자 화면에는 `Flash`, `Pro`로 표시
- 사고 모드: `빠른 응답`(thinking off), `깊은 생각`(thinking high), `최대 생각`(thinking xhigh/max)
- 웹 검색: DuckDuckGo HTML 결과를 읽는 `search_web` 함수 도구
- 세션 기억: 로컬 SQLite 파일에 대화 이력 저장
- 채팅 저장: Supabase `life_coach_sessions`, `life_coach_messages` 테이블에 사용자/코치 메시지 저장
- Google OAuth: Supabase Auth PKCE 흐름으로 로그인하고 현재 채팅 세션을 Supabase user id에 연결
- 대화 관리: 로그인 후 사이드바 `내 대화`에서 저장된 세션으로 전환, 이름 변경, 삭제
- 공유 링크: `life_coach_shared_chats`에 현재 대화 snapshot을 저장하고 `?share=...` 읽기 전용 URL로 공개
- 코칭 설정: `life_coach_user_preferences`에 답변 톤 preset과 사용자 직접 지시를 저장
- 로그인 유지: 브라우저에는 랜덤 앱 세션 토큰만 쿠키로 저장하고, refresh token은 Supabase `life_coach_auth_sessions` 테이블에 보관
- 접근 제어: owner가 있는 세션은 로그인한 owner만 로드/수정/삭제
- 공유 보안: 원본 `session=...` URL은 계속 owner 전용이고, 공유 링크는 생성 시점의 공개 snapshot만 보여줍니다.
- 코칭 스타일: `균형`, `다정`, `직설`, `실행관리`, `분석` preset 중 선택할 수 있고 직접 지시는 기본 안전 규칙보다 낮은 우선순위로 적용됩니다.
- 브라우저 표시 정리: 탭 제목/icon, Streamlit Cloud wrapper의 공유용 chrome 숨김
- 자동 응답 모드: 검색/tool-call 질문은 안정 검색 후 답변 스트리밍, 일반 대화는 바로 스트리밍
- 실시간 실행 로그: `+구간시간`, `t+누적시간` 표시
- 상태 박스: 첫 토큰 대기, 웹 검색 중, 답변 스트리밍 중 0.1초 단위 초시계 표시
- 복사 버튼: 로컬 macOS에서는 `pbcopy`로 자동 복사, 배포 환경에서는 복사용 텍스트 표시 fallback
- 실행 확인: 모델명, 응답 방식, 총 응답 시간, 검색 횟수, 검색 시간 요약 표시
- 상세 실행 정보: 검색어, 클릭 가능한 출처 링크, Runner/tool-call 타임라인 보관

일부 OpenAI-compatible provider는 일반 답변 스트리밍은 동작하지만, tool-calling 스트리밍에서 호환성 오류가 날 수 있습니다. 앱은 사용자가 토글을 고르지 않아도 자동으로 모드를 선택합니다. 웹 검색 가능성이 큰 질문은 검색 전담 agent가 `Runner.run_sync()`로 tool-call을 안정 실행한 뒤, 검색 결과를 받은 코치 agent가 `Runner.run_streamed()`로 최종 답변을 스트리밍합니다. 검색 없는 일반 대화는 바로 `Runner.run_streamed()`로 스트리밍합니다. 스트리밍 실패로 세션이 깨진 경우 새 `SQLiteSession`으로 자동 복구합니다.

## 실행 방법

이 프로젝트는 Node/Vite 앱이 아니라 Python Streamlit 앱입니다. `npm run dev`는 사용하지 않습니다.

권장 실행:

```bash
cd "/Users/sungminkim/Desktop/nomad quiz"
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
uv run --python 3.12 --with-requirements requirements.txt streamlit run app.py
```

Streamlit 첫 실행에서 이메일 입력을 물으면 빈칸으로 Enter를 눌러도 됩니다.

가상환경을 직접 만들고 싶다면, 이 머신에서는 `python3.12` 대신 `python3`를 사용하세요.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export DEEPSEEK_API_KEY="your-api-key"
python -m streamlit run app.py
```

또는 Streamlit secrets를 사용할 수 있습니다.

```toml
# .streamlit/secrets.toml
DEEPSEEK_API_KEY = "your-api-key"
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_ANON_KEY = "your-anon-or-publishable-key"
SUPABASE_SERVICE_ROLE_KEY = "your-server-side-key"
APP_BASE_URL = "http://localhost:8501"
```

`.streamlit/secrets.toml`은 커밋하지 마세요.

로컬에 `~/Documents/movie-agent/.env`가 있고 그 안에 `DEEPSEEK_API_KEY`가 있으면 자동으로 재사용합니다. 키 값은 화면에 표시하지 않습니다.

## Streamlit Cloud 배포 예습

이 과제 트랙은 Streamlit 앱이므로 Vercel/Railway보다 Streamlit Community Cloud 배포가 가장 직접적입니다.

배포 체크리스트:

1. GitHub 저장소에 `app.py`, `requirements.txt`, `README.md`, `.gitignore`를 포함합니다.
2. `.streamlit/config.toml`, `static/icons/icon-192.png`는 포함합니다.
3. 공유 링크 기능에 필요한 SQL은 `supabase/life_coach_shared_chats.sql`에 있습니다.
4. 사용자별 코칭 설정 SQL은 `supabase/life_coach_user_preferences.sql`에 있습니다.
5. `.streamlit/secrets.toml`, `.env`, `life_coach_sessions.db*`, `__pycache__/`는 커밋하지 않습니다.
6. Streamlit Community Cloud에서 새 앱을 만들고 GitHub 저장소, 브랜치, entrypoint `app.py`를 선택합니다.
7. Python version은 로컬 검증과 맞춰 `3.12`로 설정합니다.
8. Secrets에는 아래처럼 키 이름만 맞춰 등록합니다. 실제 키 값은 README나 커밋에 넣지 않습니다.

```toml
DEEPSEEK_API_KEY = "..."
SUPABASE_URL = "..."
SUPABASE_ANON_KEY = "..."
SUPABASE_SERVICE_ROLE_KEY = "..."
APP_BASE_URL = "https://your-app.streamlit.app"
```

배포 환경 메모:

- Streamlit Cloud는 Linux 환경이므로 macOS `pbcopy`가 없습니다. 복사 버튼은 자동 복사 대신 복사용 텍스트 영역을 표시합니다.
- `SQLiteSession` 파일은 Agents SDK 실행 중 컨텍스트용이고, 사용자에게 보이는 채팅 히스토리는 Supabase에 저장합니다.
- `requirements.txt`가 Python 의존성 설치 기준입니다.
- Google OAuth를 배포에서도 쓰려면 Streamlit Secrets의 `APP_BASE_URL`을 배포 URL로 바꾸고, Supabase Auth URL Configuration에도 같은 URL을 redirect allow list로 추가합니다.
- 현재 Supabase OAuth callback URL은 `https://sskejikmypcnhlcqijlr.supabase.co/auth/v1/callback`입니다. Google Cloud OAuth client의 Authorized redirect URI에 이 URL이 등록되어 있어야 합니다.

## 제출 전 체크

```bash
python -m py_compile app.py
streamlit run app.py
```
