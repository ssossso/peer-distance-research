from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import requests
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, render_template, request, redirect, session, send_file, jsonify
import random, string, json, os
from datetime import timedelta
from urllib.parse import quote, unquote
import io
import sys
import subprocess

DATABASE_URL = (os.environ.get("DATABASE_URL") or "").strip()
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = None
SessionLocal = None

if DATABASE_URL:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    SessionLocal = sessionmaker(bind=engine)

def init_db():
    if not engine:
        return

    # engine.begin(): 중간에 문제가 생기면 자동으로 롤백(되돌림)됨
    with engine.begin() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS teachers (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """))

        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS classes (
            id SERIAL PRIMARY KEY,
            code TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            teacher_username TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """))

        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS students (
            id SERIAL PRIMARY KEY,
            class_code TEXT NOT NULL,
            student_no TEXT,
            name TEXT NOT NULL
        );
        """))

        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS student_sessions (
            id SERIAL PRIMARY KEY,
            class_code TEXT NOT NULL,
            student_name TEXT NOT NULL,
            session_id TEXT NOT NULL,
            placements JSONB,
            submitted BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """))

        conn.execute(text("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_students_class_name
        ON students (class_code, name);
        """))

        # ✅ 여기부터: 교사 배치/판단 기록 테이블 추가 (반드시 init_db 안)
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS teacher_placement_runs (
            id SERIAL PRIMARY KEY,
            class_code TEXT NOT NULL,
            teacher_username TEXT NOT NULL,
            session_id TEXT NOT NULL,
            condition TEXT NOT NULL,                 -- BASELINE / TOOL_ASSISTED
            tool_run_id INTEGER,
            placements JSONB,
            started_at TIMESTAMP DEFAULT NOW(),
            ended_at TIMESTAMP,
            duration_ms INTEGER,
            confidence_score INTEGER,
            submitted BOOLEAN DEFAULT FALSE
        );
        """))

        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS teacher_decisions (
            id SERIAL PRIMARY KEY,
            run_id INTEGER NOT NULL REFERENCES teacher_placement_runs(id) ON DELETE CASCADE,
            target_student_name TEXT NOT NULL,
            priority_rank INTEGER NOT NULL,
            decision_confidence INTEGER,
            reason_tags JSONB
        );
        """))
        # ✅ 여기까지

                # ✅ 분석 결과 캐시 (요청 시 계산 + 캐시)
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS analysis_cache (
            id SERIAL PRIMARY KEY,
            class_code TEXT NOT NULL,
            session_id TEXT NOT NULL,
            cache_key TEXT NOT NULL,      -- 예: avg_map, teacher_run_123, student_avg_dist
            payload JSONB,
            updated_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(class_code, session_id, cache_key)
        );
        """))

    # (선택) 명시적으로 종료를 드러내고 싶다면 return을 둬도 됩니다.
    return


def db_get_student_session(class_code: str, student_name: str, sid: str):
    """
    학생 1명의 특정 회차 세션 가져오기
    return: {"placements": dict, "submitted": bool} 또는 None
    """
    if not engine:
        return None


    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT placements, submitted
            FROM student_sessions
            WHERE class_code = :code
              AND student_name = :name
              AND session_id = :sid
            LIMIT 1
        """), {
            "code": class_code,
            "name": student_name,
            "sid": sid
        }).fetchone()

    if not row:
        return None

    return {
        "placements": row.placements or {},
        "submitted": bool(row.submitted)
    }

def db_delete_class_for_teacher(class_code: str, teacher_username: str) -> bool:
    """
    교사 본인 소유의 학급만 삭제. 관련 데이터(students, student_sessions)도 함께 삭제.
    성공 True / 실패 False
    """
    if not engine:
        raise RuntimeError("DB engine not initialized")

    class_code = (class_code or "").upper().strip()

    with engine.begin() as conn:
        # 권한 확인
        row = conn.execute(text("""
            SELECT 1
            FROM classes
            WHERE code = :code AND teacher_username = :t
            LIMIT 1
        """), {"code": class_code, "t": teacher_username}).fetchone()

        if not row:
            return False

        # 자식 테이블부터 삭제 (FK가 없어도 안전)
        conn.execute(text("DELETE FROM student_sessions WHERE class_code = :code"), {"code": class_code})
        conn.execute(text("DELETE FROM students WHERE class_code = :code"), {"code": class_code})

        # 교사 배치 관련(있으면 같이 삭제)
        conn.execute(text("DELETE FROM teacher_placement_runs WHERE class_code = :code"), {"code": class_code})

        # 마지막으로 classes 삭제
        conn.execute(text("DELETE FROM classes WHERE code = :code AND teacher_username = :t"), {
            "code": class_code,
            "t": teacher_username
        })

    return True


def db_create_class(teacher_username: str, class_code: str, class_name: str, students: list[dict]):
    """
    classes, students 테이블에 학급/학생 저장
    students: [{"no": "1", "name": "홍길동"}, ...]
    """
    if not engine:
        raise RuntimeError("DB engine not initialized")

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO classes (code, name, teacher_username)
            VALUES (:code, :name, :teacher_username)
        """), {
            "code": class_code,
            "name": class_name,
            "teacher_username": teacher_username,
        })

        for s in students:
            conn.execute(text("""
                INSERT INTO students (class_code, student_no, name)
                VALUES (:class_code, :student_no, :name)
            """), {
                "class_code": class_code,
                "student_no": str(s.get("no", "")),
                "name": (s.get("name") or "").strip(),
            })


def db_list_classes_for_teacher(teacher_username: str) -> dict:
    """
    대시보드용: 교사 계정이 가진 학급 목록을 dict로 반환
    기존 dashboard.html이 기대하는 구조를 최대한 맞춰줌:
    {code: {"name": ..., "teacher": ...}, ...}
    """
    if not engine:
        raise RuntimeError("DB engine not initialized")

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT code, name, teacher_username
            FROM classes
            WHERE teacher_username = :t
            ORDER BY id DESC
        """), {"t": teacher_username}).fetchall()

    out = {}
    for r in rows:
        out[r.code] = {"name": r.name, "teacher": r.teacher_username}
    return out

def db_get_class_name(class_code: str):
    if not engine:
        return None

    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT name
            FROM classes
            WHERE code = :code
            LIMIT 1
        """), {"code": class_code}).fetchone()

    return row.name if row else None



def db_get_class_for_teacher(class_code: str, teacher_username: str):
    """
    교사 권한 확인 포함해서 학급 1개 가져오기
    """
    if not engine:
        raise RuntimeError("DB engine not initialized")

    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT code, name, teacher_username
            FROM classes
            WHERE code = :code
        """), {"code": class_code}).fetchone()

    if not row:
        return None
    if row.teacher_username != teacher_username:
        return "FORBIDDEN"

    return {"code": row.code, "name": row.name, "teacher": row.teacher_username, "sessions": {}}


def db_get_students_in_class(class_code: str):
    """
    학급의 학생 목록 가져오기 (번호 포함)
    """
    if not engine:
        raise RuntimeError("DB engine not initialized")

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT student_no, name
            FROM students
            WHERE class_code = :code
            ORDER BY id ASC
        """), {"code": class_code}).fetchall()

    out = []
    for r in rows:
        out.append({"no": (r.student_no or ""), "name": r.name})
    return out


def db_get_submitted_map(class_code: str, sid: str):
    """
    해당 학급/회차에서 학생별 제출 여부 맵:
    {"홍길동": True, "김철수": False, ...}
    """
    if not engine:
        raise RuntimeError("DB engine not initialized")

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT student_name, submitted
            FROM student_sessions
            WHERE class_code = :code AND session_id = :sid
        """), {"code": class_code, "sid": sid}).fetchall()

    m = {}
    for r in rows:
        m[r.student_name] = bool(r.submitted)
    return m

def db_list_submitted_student_sessions(class_code: str, sid: str):
    """
    해당 학급/회차에서 제출(submitted=True)한 학생들의 placements를 모두 가져오기
    return: [{"student_name": ..., "placements": {...}}, ...]
    """
    if not engine:
        return []

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT student_name, placements
            FROM student_sessions
            WHERE class_code = :code
              AND session_id = :sid
              AND submitted = TRUE
            ORDER BY id ASC
        """), {"code": class_code, "sid": sid}).fetchall()

    out = []
    for r in rows:
        out.append({"student_name": r.student_name, "placements": r.placements or {}})
    return out


def db_upsert_student_session(class_code: str, student_name: str, sid: str, placements: dict, submitted: bool):
    """
    student_sessions에 (학급코드, 학생이름, 회차) 기준으로 저장/갱신
    """
    if not engine:
        raise RuntimeError("DB engine not initialized")

    with engine.begin() as conn:
        # 이미 있으면 UPDATE, 없으면 INSERT
        row = conn.execute(text("""
            SELECT id FROM student_sessions
            WHERE class_code = :code AND student_name = :name AND session_id = :sid
            LIMIT 1
        """), {"code": class_code, "name": student_name, "sid": sid}).fetchone()

        if row:
            conn.execute(text("""
                UPDATE student_sessions
                SET placements = :placements, submitted = :submitted
                WHERE id = :id
            """), {
                "placements": json.dumps(placements, ensure_ascii=False),
                "submitted": submitted,
                "id": row.id
            })
        else:
            conn.execute(text("""
                INSERT INTO student_sessions (class_code, student_name, session_id, placements, submitted)
                VALUES (:code, :name, :sid, :placements, :submitted)
            """), {
                "code": class_code,
                "name": student_name,
                "sid": sid,
                "placements": json.dumps(placements, ensure_ascii=False),
                "submitted": submitted
            })

def db_create_teacher_run(class_code: str, teacher_username: str, sid: str, condition: str, tool_run_id=None) -> int:
    if not engine:
        raise RuntimeError("DB engine not initialized")

    with engine.begin() as conn:
        row = conn.execute(text("""
            INSERT INTO teacher_placement_runs (class_code, teacher_username, session_id, condition, tool_run_id, placements, submitted)
            VALUES (:code, :t, :sid, :cond, :tool_run_id, :placements, FALSE)
            RETURNING id
        """), {
            "code": class_code,
            "t": teacher_username,
            "sid": sid,
            "cond": condition,
            "tool_run_id": tool_run_id,
            "placements": json.dumps({}, ensure_ascii=False),
        }).fetchone()
    return int(row.id)


def db_get_teacher_run(run_id: int):
    if not engine:
        return None

    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT id, class_code, teacher_username, session_id, condition, tool_run_id,
                   placements, submitted, started_at, ended_at, duration_ms, confidence_score
            FROM teacher_placement_runs
            WHERE id = :id
            LIMIT 1
        """), {"id": run_id}).fetchone()

    if not row:
        return None

    return {
        "id": row.id,
        "class_code": row.class_code,
        "teacher_username": row.teacher_username,
        "session_id": row.session_id,
        "condition": row.condition,
        "tool_run_id": row.tool_run_id,
        "placements": row.placements or {},
        "submitted": bool(row.submitted),
        "started_at": row.started_at,
        "ended_at": row.ended_at,
        "duration_ms": row.duration_ms,
        "confidence_score": row.confidence_score,
    }


def db_update_teacher_run_placements(run_id: int, placements: dict):
    if not engine:
        raise RuntimeError("DB engine not initialized")

    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE teacher_placement_runs
            SET placements = :placements
            WHERE id = :id
        """), {
            "placements": json.dumps(placements, ensure_ascii=False),
            "id": run_id
        })


def db_complete_teacher_run(run_id: int, duration_ms: int, confidence_score: int):
    if not engine:
        raise RuntimeError("DB engine not initialized")

    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE teacher_placement_runs
            SET ended_at = NOW(),
                duration_ms = :duration_ms,
                confidence_score = :confidence_score,
                submitted = TRUE
            WHERE id = :id
        """), {
            "duration_ms": duration_ms,
            "confidence_score": confidence_score,
            "id": run_id
        })


def db_replace_teacher_decisions(run_id: int, decisions: list[dict]):
    """결정(우선순위)은 수정 가능성이 있으니, 저장 시 기존 것 삭제 후 다시 넣는 방식이 단순합니다."""
    if not engine:
        raise RuntimeError("DB engine not initialized")

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM teacher_decisions WHERE run_id = :run_id"), {"run_id": run_id})

        for d in decisions:
            conn.execute(text("""
                INSERT INTO teacher_decisions (run_id, target_student_name, priority_rank, decision_confidence, reason_tags)
                VALUES (:run_id, :name, :rank, :conf, :tags)
            """), {
                "run_id": run_id,
                "name": (d.get("name") or "").strip(),
                "rank": int(d.get("rank") or 0),
                "conf": int(d.get("confidence") or 0) if d.get("confidence") else None,
                "tags": json.dumps(d.get("tags"), ensure_ascii=False) if d.get("tags") is not None else None,
            })



# 서버 시작 시 DB 테이블 자동 생성
try:
    init_db()
except Exception as e:
    print("init_db failed:", e)


app = Flask(__name__)

# (진단용) Internal Server Error 원인을 화면/로그에 더 잘 보이게
app.config["PROPAGATE_EXCEPTIONS"] = True

# Render 환경변수에 SECRET_KEY를 넣고 고정해야 배포해도 로그인 유지가 됩니다.
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-change-me")

# 세션(로그인 상태)을 30일 유지
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=30)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

@app.before_request
def make_session_permanent():
    session.permanent = True


SITE_TITLE = "내가 바라본 우리 반"
# Render Persistent Disk 경로 사용 (기본값: /var/data/data.json)
DATA_FILE = os.environ.get("DATA_FILE", "data.json")


# --- Google Sheets 연동 ---
GOOGLE_WEBAPP_URL = "https://script.google.com/macros/s/AKfycbwyjKC2JearJnySkxdG0oahMkMJ5V6uBqY5EYRGVVRa8KWZvRzHcskeVNY5hnlyiSw/exec"
GOOGLE_SECRET = os.environ.get("GOOGLE_SECRET", "").strip()

DEBUG_MODE = os.environ.get("DEBUG_MODE") == "1"


# ---------- Google Sheets POST ----------
def post_to_sheet(payload: dict) -> dict:
    payload = dict(payload)
    payload["secret"] = GOOGLE_SECRET

    try:
        r = requests.post(GOOGLE_WEBAPP_URL, json=payload, timeout=10)
    except Exception as e:
        # 네트워크 오류 등
        return {
            "status": "error",
            "message": f"request failed: {e}"
        }

    # HTTP 에러 상태 코드
    if r.status_code != 200:
        return {
            "status": "error",
            "message": f"http {r.status_code}",
            "text": r.text[:300]
        }

    # JSON 파싱 안전 처리
    try:
        return r.json()
    except Exception:
        return {
            "status": "error",
            "message": "invalid json response",
            "text": r.text[:300]
        }


# ---------- 데이터 ----------
def load_data():
    # 파일이 없으면 빈 데이터
    if not os.path.exists(DATA_FILE):
        return {"classes": {}}

    # 파일이 있어도, JSON이 깨져 있으면 앱이 죽지 않게 방어
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        # 깨진 파일이면 일단 비워서라도 서비스가 살아있게 함
        return {"classes": {}}

    # 데이터 구조 안전 보정
    d.setdefault("classes", {})
    for code, cls in d.get("classes", {}).items():
        d["classes"][code] = ensure_class_schema(cls)

    return d


def save_data(data):
    # /var/data 같은 경로는 폴더가 없을 수 있으니 먼저 생성
    parent_dir = os.path.dirname(DATA_FILE) or "."
    os.makedirs(parent_dir, exist_ok=True)

    # 안전 저장: tmp 파일에 먼저 쓰고 마지막에 교체
    tmp_path = DATA_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    os.replace(tmp_path, DATA_FILE)


def save_data_safely(d):
    for code, cls in d.get("classes", {}).items():
        d["classes"][code] = ensure_class_schema(cls)
    save_data(d)

def make_code():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

def ensure_class_schema(cls):
    if not cls:
        return cls
    cls.setdefault("sessions", {})
    for i in range(1, 6):
        cls["sessions"].setdefault(str(i), {"label": f"{i}차", "active": i == 1})

    for name, sdata in cls.get("students_data", {}).items():
        sdata.setdefault("sessions", {})
        for sid in cls["sessions"]:
            sdata["sessions"].setdefault(sid, {"placements": {}, "submitted": False})
    return cls

def get_current_class():
    """
    상단바에 표시할 '현재 선택 학급'을 계산합니다.
    - 교사: session['selected_class']가 있으면 그 학급
    - 학생: session['code']가 있으면 그 학급
    """
    code = None

    if "teacher" in session and session.get("selected_class"):
        code = session.get("selected_class")
    elif session.get("code"):
        code = session.get("code")

    if not code:
        return None

    # ✅ DB 우선
    if engine:
        name = db_get_class_name(code)
        if not name:
            return None
        return {"name": name, "code": code}

    # (예외) DB 없을 때만 JSON
    d = load_data()
    cls = d.get("classes", {}).get(code)
    if not cls:
        return None
    return {"name": cls.get("name", ""), "code": code}



@app.context_processor
def inject_globals():
    return {"current_class": get_current_class()}

# ---------- DB 연결 테스트용 임시 라우트 ----------
@app.route("/debug/db")
def debug_db():
    if not engine:
        return "DATABASE_URL not set"

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return "DB connection OK"
    except Exception as e:
        return f"DB connection failed: {e}", 500


# ---------- 홈 ----------
@app.route("/")
def home():
    return render_template("home.html", site_title=SITE_TITLE)

# ---------- 교사 회원가입 ----------
@app.route("/teacher/signup", methods=["GET", "POST"])
def teacher_signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        pw = request.form.get("password", "")
        pw2 = request.form.get("password2", "")

        if not username or not pw:
            return render_template("teacher_signup.html", error="아이디/비밀번호를 입력해 주세요.")
        if pw != pw2:
            return render_template("teacher_signup.html", error="비밀번호가 서로 다릅니다.")

        pw_hash = generate_password_hash(pw)

        try:
            resp = post_to_sheet({
                "action": "teacher_signup",
                "username": username,
                "pw_hash": pw_hash
            })
        except Exception as e:
            return render_template("teacher_signup.html", error=f"서버 통신 오류: {e}")

        # 디버그: 응답 status를 그대로 보여줌
        status = resp.get("status")
        if status == "ok":
            return redirect("/teacher/login")

        if status == "exists":
            return render_template("teacher_signup.html", error="이미 존재하는 아이디입니다.")

        if status == "blocked":
            return render_template("teacher_signup.html", error="blocked: GOOGLE_SECRET(비밀키) 불일치 또는 누락")

        # error 등 기타
        return render_template("teacher_signup.html", error=f"회원가입 실패: {resp}")

    return render_template("teacher_signup.html")


# ---------- 교사 로그인 ----------
@app.route("/teacher/login", methods=["GET", "POST"])
def teacher_login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        pw = request.form.get("password", "")

        try:
            resp = post_to_sheet({
                "action": "teacher_get",
                "username": username
            })
        except Exception as e:
            return render_template("teacher_login.html", error=f"서버 통신 오류: {e}")

        try:
            if resp.get("status") != "ok":
                return render_template("teacher_login.html", error=f"로그인 실패: {resp}")

            pw_hash = resp.get("pw_hash") or ""
            if check_password_hash(pw_hash, pw):
                session.clear()
                session["teacher"] = username
                return redirect("/teacher/dashboard")

            return render_template("teacher_login.html", error="로그인 실패: 비밀번호 불일치")
        except Exception as e:
            # 여기서 터지면 어떤 값 때문에 터졌는지 노출
            return render_template("teacher_login.html", error=f"로그인 처리 중 오류: {e} / resp={resp}")

    return render_template("teacher_login.html")


# ---------- 임시디버그(스위치로 ON/OFF) ----------
if DEBUG_MODE:
    import hashlib

    @app.route("/debug/secret")
    def debug_secret():
        s = GOOGLE_SECRET or ""
        return {
            "len": len(s),
            "sha256_8": hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]
        }

    @app.route("/debug/sheets")
    def debug_sheets():
        return {
            "webapp_url": GOOGLE_WEBAPP_URL,
            "secret_len": len(GOOGLE_SECRET or ""),
            "secret_sha256_8": hashlib.sha256((GOOGLE_SECRET or "").encode("utf-8")).hexdigest()[:8],
        }

# ---------- 로그아웃 ----------
@app.route("/teacher/logout")
def teacher_logout():
    session.clear()
    return redirect("/")

# ---------- 교사 대시보드 ----------
@app.route("/teacher/dashboard")
def dashboard():
    if "teacher" not in session:
        return redirect("/teacher/login")

    # ✅ classes는 무조건 먼저 만들어 둠(빈 dict라도)
    classes = {}

    # ✅ DB가 있으면 DB에서 읽기, DB가 없거나 실패하면 JSON로 fallback
    try:
        if engine:
            classes = db_list_classes_for_teacher(session["teacher"])
        else:
            d = load_data()
            classes = {c: v for c, v in d.get("classes", {}).items() if v.get("teacher") == session["teacher"]}
    except Exception as e:
        # 서비스가 죽지 않게 fallback
        d = load_data()
        classes = {c: v for c, v in d.get("classes", {}).items() if v.get("teacher") == session["teacher"]}

    # (추가) 현재 선택 학급이 없으면, 첫 번째 학급을 자동 선택해서 상단바가 뜨게 함
    if classes and not session.get("selected_class"):
        first_code = next(iter(classes.keys()))
        session["selected_class"] = first_code

    return render_template("dashboard.html", classes=classes)


# ---------- 학급 생성 ----------
@app.route("/teacher/create", methods=["GET", "POST"])
def create_class():
    if "teacher" not in session:
        return redirect("/teacher/login")

    if request.method == "POST":
        d = load_data()
        code = make_code()

        class_name = request.form.get("class_name", "").strip()
        students_raw = request.form.get("students", "")

        parsed = []
        auto_no = 1
        for line in students_raw.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split("\t")]
            if len(parts) == 1:
                parts = [p.strip() for p in line.split(",")]
            name = parts[-1]
            parsed.append({"no": str(auto_no), "name": name})
            auto_no += 1

        # DB에 학급 생성 저장
        db_create_class(
            teacher_username=session["teacher"],
            class_code=code,
            class_name=class_name or f"학급 {code}",
            students=parsed,
        )

        # 대시보드 상단바 표시용으로 "선택 학급"도 세션에 넣어두기
        session["selected_class"] = code

        return redirect("/teacher/dashboard")


    return render_template("create_class.html")

# ---------- 학급 삭제 ----------
@app.route("/teacher/class/delete", methods=["POST"])
def teacher_delete_class():
    if "teacher" not in session:
        return redirect("/teacher/login")

    code = (request.form.get("code") or "").upper().strip()
    if not code:
        return redirect("/teacher/dashboard")

    try:
        ok = db_delete_class_for_teacher(code, session["teacher"])
        # 실패해도 대시보드로 복귀(권한 없거나 이미 삭제된 경우)
        return redirect("/teacher/dashboard")
    except Exception as e:
        # 필요하면 dashboard에 에러 표시로 확장 가능
        return redirect("/teacher/dashboard")


# ---------- 학급 상세 ----------
@app.route("/teacher/class/<code>")
def class_detail(code):
    if "teacher" not in session:
        return redirect("/teacher/login")

    code = (code or "").upper().strip()

    # 회차 선택(쿼리스트링 우선) - 기존 로직 유지
    sid = (request.args.get("sid") or session.get("selected_session") or "1").strip()
    if sid not in ["1", "2", "3", "4", "5"]:
        sid = "1"
    session["selected_session"] = sid

    # ✅ DB가 있으면 DB에서 학급 조회
    if engine:
        cls = db_get_class_for_teacher(code, session["teacher"])
        if cls == "FORBIDDEN":
            return "학급을 찾을 수 없거나 접근 권한이 없습니다.", 404
        if not cls:
            return "학급을 찾을 수 없거나 접근 권한이 없습니다.", 404

        # 세션 메타(기존 화면 유지용)
        cls = ensure_class_schema(cls)

        # 학생 목록
        students = db_get_students_in_class(code)
        cls["students"] = students

        # 제출 상태 맵
        submitted_map = db_get_submitted_map(code, sid)

        # 상단바 표시용
        session["selected_class"] = code

        # 학생 목록 + 상태 (기존 템플릿이 기대하는 rows 형태 유지)
        rows = []
        for i, item in enumerate(students, start=1):
            no = str(item.get("no", "") or i)
            name = (item.get("name") or "").strip()
            if not name:
                continue

            submitted = bool(submitted_map.get(name, False))
            status = "완료" if submitted else "미완료"
            rows.append({"no": no, "name": name, "status": status, "url_name": quote(name)})

        # 회차 링크
        session_links = []
        for _sid, meta in sorted(cls.get("sessions", {}).items(), key=lambda x: int(x[0])):
            session_links.append({
                "sid": _sid,
                "label": meta.get("label", f"{_sid}차"),
                "url": f"/s/{code}/{_sid}",
            })

        return render_template(
            "class_detail.html",
            cls=cls,
            code=code,
            rows=rows,
            sid=sid,
            session_links=session_links,
        )

    # ✅ DB가 없으면(예외적) 기존 JSON 방식 fallback
    d = load_data()
    cls = ensure_class_schema(d.get("classes", {}).get(code))
    if not cls or cls.get("teacher") != session["teacher"]:
        return "학급을 찾을 수 없거나 접근 권한이 없습니다.", 404

    session["selected_class"] = code

    rows = []
    for i, item in enumerate(cls.get("students", []), start=1):
        if isinstance(item, dict):
            no = str(item.get("no", "") or i)
            name = (item.get("name") or "").strip()
        else:
            no = str(i)
            name = (item or "").strip()

        if not name:
            continue

        submitted = bool(
            cls.get("students_data", {})
              .get(name, {})
              .get("sessions", {})
              .get(sid, {})
              .get("submitted", False)
        )
        status = "완료" if submitted else "미완료"
        rows.append({"no": no, "name": name, "status": status, "url_name": quote(name)})

    session_links = []
    for _sid, meta in sorted(cls.get("sessions", {}).items(), key=lambda x: int(x[0])):
        session_links.append({
            "sid": _sid,
            "label": meta.get("label", f"{_sid}차"),
            "url": f"/s/{code}/{_sid}",
        })

    return render_template(
        "class_detail.html",
        cls=cls,
        code=code,
        rows=rows,
        sid=sid,
        session_links=session_links,
    )

@app.route("/teacher/class/<code>/result/<sid>/<url_name>")
def teacher_view_student_result(code, sid, url_name):
    if "teacher" not in session:
        return redirect("/teacher/login")

    code = (code or "").upper().strip()
    sid = (sid or "1").strip()
    if sid not in ["1","2","3","4","5"]:
        sid = "1"

    student_name = (unquote(url_name) or "").strip()

    # 권한 확인 (해당 학급이 이 교사의 것인지)
    cls = db_get_class_for_teacher(code, session["teacher"])
    if cls == "FORBIDDEN" or not cls:
        return "학급을 찾을 수 없거나 접근 권한이 없습니다.", 404

    # ✅ student_write.html이 필요로 하는 friends 만들기
    students = db_get_students_in_class(code)
    all_names = [s["name"] for s in students]

    if student_name not in all_names:
        return "학생을 찾을 수 없습니다.", 404

    friends = [n for n in all_names if n != student_name]

    # ✅ DB에서 학생 세션 가져오기
    sess = db_get_student_session(code, student_name, sid)
    placements = (sess.get("placements") if sess else {}) or {}

    # ✅ 교사 열람은 무조건 읽기 전용으로 잠금
    # student_write.html은 student_session.submitted 값으로 READONLY를 결정하는 구조(당신 teacher_write.html도 동일)
    student_session = {"placements": placements, "submitted": True}

    # ✅ session_meta(라벨) 제공: student_write()가 하는 방식과 동일하게 ensure_class_schema 사용
    cls_for_view = ensure_class_schema({
        "code": code,
        "name": db_get_class_name(code) or code,
        "teacher": session["teacher"],
        "sessions": {}
    })
    session_meta = cls_for_view.get("sessions", {}).get(sid, {"label": f"{sid}차"})

    # ✅ 기존 학생 화면을 그대로 렌더링
    # (템플릿 새로 만들지 않음)
    return render_template(
        "student_write.html",
        name=student_name,
        friends=friends,
        placements=placements,
        student_session=student_session,
        sid=sid,
        session_meta=session_meta,
        teacher_view=True  # (선택) 템플릿에서 '교사용 열람' 문구 표시 용도
    )



@app.route("/s/<code>/<sid>", methods=["GET", "POST"])
def student_enter_session(code, sid):
    """
    교사가 회차별 링크(/s/<code>/<sid>)를 배포하면
    학생은 회차 선택 없이 바로 해당 회차로 입장합니다.
    """
    code = (code or "").upper().strip()
    sid = (sid or "1").strip()

    # ✅ DB 우선: 학급 존재 확인 + 학생 명단 로드
    if engine:
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT code, name
                FROM classes
                WHERE code = :code
                LIMIT 1
            """), {"code": code}).fetchone()

        if not row:
            return "학급을 찾을 수 없습니다.", 404

        # 화면용 sessions 메타(기존 템플릿 유지)
        cls = ensure_class_schema({"code": code, "name": row.name, "sessions": {}})

        # DB 학생 목록 -> 템플릿이 쓰는 형태로 맞춤
        students = db_get_students_in_class(code)
        cls["students"] = students
        cls["students_data"] = {s["name"]: {"sessions": {}} for s in students}

    else:
        # (예외) DB 없을 때만 JSON
        d = load_data()
        cls = ensure_class_schema(d.get("classes", {}).get(code))
        if not cls:
            return "학급을 찾을 수 없습니다.", 404


    if sid not in cls.get("sessions", {}):
        sid = "1"


    if request.method == "POST":
        name = request.form.get("name", "").strip()
        if name not in cls.get("students_data", {}):
            return render_template(
                "student_enter_session.html",
                error="학생 명단에 없는 이름입니다.",
                code=code,
                sid=sid,
                session_label=cls.get("sessions", {}).get(sid, {}).get("label", f"{sid}차"),
            )

        session["code"] = code
        session["name"] = name
        session["sid"] = sid

        # 상단바 표시용(선택 학급/회차)
        session["selected_class"] = code
        session["selected_session"] = sid

        return redirect("/student/write")

    return render_template(
        "student_enter_session.html",
        code=code,
        sid=sid,
        session_label=cls.get("sessions", {}).get(sid, {}).get("label", f"{sid}차"),
    )

@app.route("/qr/<code>/<sid>.png")
def qr_session_link(code, sid):
    code = (code or "").upper().strip()
    sid = str(sid).strip()
    if sid not in ["1", "2", "3", "4", "5"]:
        sid = "1"

    # ✅ DB 우선: 학급 존재만 확인하면 됨
    if engine:
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT 1 FROM classes WHERE code = :code LIMIT 1
            """), {"code": code}).fetchone()
        if not row:
            return "학급을 찾을 수 없습니다.", 404
    else:
        # (예외) DB 없을 때만 JSON 확인
        d = load_data()
        cls = d.get("classes", {}).get(code)
        if not cls:
            return "학급을 찾을 수 없습니다.", 404
        cls = ensure_class_schema(cls)

    base = request.url_root.rstrip("/")
    target = f"{base}/s/{code}/{sid}"

    try:
        import qrcode
    except ModuleNotFoundError:
        return "QR 코드 생성을 위해 qrcode 라이브러리가 필요합니다.", 500

    img = qrcode.make(target)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    resp = send_file(buf, mimetype="image/png")
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp

@app.route("/teacher/class/<code>/placement/start")
def teacher_placement_start(code):
    if "teacher" not in session:
        return redirect("/teacher/login")

    code = (code or "").upper().strip()
    sid = (request.args.get("sid") or session.get("selected_session") or "1").strip()
    if sid not in ["1","2","3","4","5"]:
        sid = "1"

    condition = (request.args.get("condition") or "BASELINE").strip()
    if condition not in ["BASELINE", "TOOL_ASSISTED"]:
        condition = "BASELINE"

    tool_run_id = request.args.get("tool_run_id")
    tool_run_id = int(tool_run_id) if (tool_run_id and tool_run_id.isdigit()) else None

    run_id = db_create_teacher_run(code, session["teacher"], sid, condition, tool_run_id=tool_run_id)
    return redirect(f"/teacher/placement/{run_id}")

@app.route("/teacher/placement/<int:run_id>", methods=["GET", "POST"])
def teacher_placement_write(run_id):
    if "teacher" not in session:
        return redirect("/teacher/login")

    run = db_get_teacher_run(run_id)
    if not run or run["teacher_username"] != session["teacher"]:
        return "접근 권한이 없습니다.", 403

    code = run["class_code"]
    sid = run["session_id"]

    students = db_get_students_in_class(code)
    all_names = [s["name"] for s in students]  # 교사 화면에서는 전체가 '대상'
    placements = run["placements"] or {}

    if request.method == "POST":
        placements_json = (request.form.get("placements_json") or "{}").strip()
        try:
            placements_obj = json.loads(placements_json) if placements_json else {}
        except Exception:
            placements_obj = {}

        db_update_teacher_run_placements(run_id, placements_obj)

        # 배치 저장 후 완료(설문) 화면으로 이동
        return redirect(f"/teacher/placement/{run_id}/complete")

    return render_template(
        "teacher_write.html",
        run=run,
        code=code,
        sid=sid,
        friends=all_names,
        placements=placements,
    )

@app.route("/teacher/placement/<int:run_id>/complete", methods=["GET", "POST"])
def teacher_placement_complete(run_id):
    if "teacher" not in session:
        return redirect("/teacher/login")

    run = db_get_teacher_run(run_id)
    if not run or run["teacher_username"] != session["teacher"]:
        return "접근 권한이 없습니다.", 403

    if request.method == "POST":
        duration_ms = request.form.get("duration_ms") or "0"
        confidence_score = request.form.get("confidence_score") or "0"

        try:
            duration_ms = int(duration_ms)
        except Exception:
            duration_ms = 0

        try:
            confidence_score = int(confidence_score)
        except Exception:
            confidence_score = 0

        # 우선순위(예: 1~3위) - form에서 name="priority_1" ... 형태로 받는 방식 추천
        decisions = []
        for rank in [1,2,3]:
            nm = (request.form.get(f"priority_{rank}") or "").strip()
            if nm:
                decisions.append({"name": nm, "rank": rank})

        db_replace_teacher_decisions(run_id, decisions)
        db_complete_teacher_run(run_id, duration_ms=duration_ms, confidence_score=confidence_score)

        # 완료 후 학급 상세로 이동
        return redirect(f"/teacher/class/{run['class_code']}?sid={run['session_id']}")

    return render_template("teacher_complete.html", run=run)


@app.route("/analysis/class/<code>/<sid>/teacher_run/<int:run_id>.json")
def analysis_teacher_run(code, sid, run_id):
    if "teacher" not in session:
        return redirect("/teacher/login")

    code = (code or "").upper().strip()
    sid = (sid or "1").strip()

    run = db_get_teacher_run(run_id)
    if (not run) or run["class_code"] != code or run["session_id"] != sid:
        return jsonify({"error": "run not found"}), 404
    if run["teacher_username"] != session["teacher"]:
        return jsonify({"error": "forbidden"}), 403

    cache_key = f"teacher_run_{run_id}"
    cached = cache_get(code, sid, cache_key)
    if cached:
        return jsonify(cached)

    payload = teacher_run_distance_payload(code, sid, run_id)
    if not payload:
        return jsonify({"error": "failed"}), 500

    cache_set(code, sid, cache_key, payload)
    return jsonify(payload)

@app.route("/analysis/class/<code>/<sid>/student_avg.json")
def analysis_student_avg(code, sid):
    if "teacher" not in session:
        return redirect("/teacher/login")

    code = (code or "").upper().strip()
    sid = (sid or "1").strip()
    if sid not in ["1","2","3","4","5"]:
        sid = "1"

    # 교사 권한 확인: 해당 학급 소유자만 조회 가능
    cls = db_get_class_for_teacher(code, session["teacher"])
    if cls == "FORBIDDEN" or not cls:
        return jsonify({"error": "forbidden"}), 403

    cache_key = f"student_avg_{sid}"
    cached = cache_get(code, sid, cache_key)
    if cached:
        return jsonify(cached)

    payload = student_avg_distance_payload(code, sid)
    cache_set(code, sid, cache_key, payload)
    return jsonify(payload)


def points_from_student_session(placements: dict, names: list[str], self_name: str):
    """
    학생 1명(=응답자)의 배치를 전체 학생명(names, 길이 N)에 맞춰 point list로 변환.
    - self_name(응답자)은 placements에 없으므로 (0,0)으로 암묵 포함
    - 나머지는 placements[name].x, placements[name].y 사용 (student_write의 rel 좌표)
    - 누락은 invalid 처리
    - 스케일 차이를 줄이기 위해 bbox 기반 0~1 정규화(방향은 의미 없고 거리만 쓰므로 안전)
    """
    pts = []
    valid = []

    for nm in names:
        if nm == self_name:
            pts.append((0.0, 0.0))
            valid.append(True)
            continue

        v = placements.get(nm)
        if not isinstance(v, dict):
            pts.append((0.0, 0.0))
            valid.append(False)
            continue

        x = v.get("x")
        y = v.get("y")
        if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
            pts.append((0.0, 0.0))
            valid.append(False)
            continue

        pts.append((float(x), float(y)))
        valid.append(True)

    # bbox 정규화 (valid 포인트들 기준)
    xs = [pts[i][0] for i in range(len(pts)) if valid[i]]
    ys = [pts[i][1] for i in range(len(pts)) if valid[i]]
    if len(xs) >= 2 and len(ys) >= 2:
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        rx = (maxx - minx) if (maxx - minx) > 1e-9 else 1.0
        ry = (maxy - miny) if (maxy - miny) > 1e-9 else 1.0
        pts = [((x - minx)/rx, (y - miny)/ry) for (x, y) in pts]

    return pts, valid


def student_avg_distance_payload(class_code: str, sid: str):
    """
    학생 제출 전체(submitted=True)로부터:
    - 학생별 거리행렬 생성
    - pairwise mean으로 평균 거리행렬 생성
    - MDS로 평균맵 좌표 생성
    """
    students = db_get_students_in_class(class_code)
    names = [s["name"] for s in students]
    n_total = len(names)
    if n_total == 0:
        return {
            "class_code": class_code,
            "session_id": sid,
            "names": [],
            "n_submitted": 0,
            "avg_distance_matrix": [],
            "mds_2d": [],
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }

    submitted = db_list_submitted_student_sessions(class_code, sid)

    mats = []
    used_students = []
    for item in submitted:
        self_name = item.get("student_name")
        placements = item.get("placements") or {}

        if self_name not in names:
            continue

        pts, valid = points_from_student_session(placements, names, self_name=self_name)
        D = distance_matrix(pts, valid)
        mats.append(D)
        used_students.append(self_name)

    avgD = mean_distance_matrix(mats)
    X = classical_mds_2d(avgD) if avgD else []

    return {
        "class_code": class_code,
        "session_id": sid,
        "names": names,
        "n_total": n_total,
        "n_submitted": len(used_students),
        "submitted_students": used_students,
        "avg_distance_matrix": avgD,
        "mds_2d": [{"x": X[i][0], "y": X[i][1]} for i in range(len(X))],
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }



# ---------- 학생 입장 ----------
@app.route("/student", methods=["GET", "POST"])
def student_enter():
    if request.method == "POST":
        code = request.form.get("code", "").upper().strip()
        name = request.form.get("name", "").strip()

        # ✅ 1) DB 우선
        if engine:
            # 학급 존재 확인
            with engine.connect() as conn:
                c_row = conn.execute(text("""
                    SELECT code
                    FROM classes
                    WHERE code = :code
                    LIMIT 1
                """), {"code": code}).fetchone()

            if not c_row:
                return render_template("student_enter.html", error="입장 실패")

            # 학생 존재 확인
            with engine.connect() as conn:
                s_row = conn.execute(text("""
                    SELECT 1
                    FROM students
                    WHERE class_code = :code AND name = :name
                    LIMIT 1
                """), {"code": code, "name": name}).fetchone()

            if not s_row:
                return render_template("student_enter.html", error="입장 실패")

            # 세션 저장
            session["code"] = code
            session["name"] = name
            session["sid"] = "1"

            # 상단바 표시용(선택 학급/회차)
            session["selected_class"] = code
            session["selected_session"] = "1"

            return redirect("/student/write")

        # ✅ 2) (예외) DB가 없으면 JSON 방식 fallback
        d = load_data()
        cls = d.get("classes", {}).get(code)
        if not cls or name not in (cls.get("students_data") or {}):
            return render_template("student_enter.html", error="입장 실패")

        session["code"] = code
        session["name"] = name
        session["sid"] = "1"
        session["selected_class"] = code
        session["selected_session"] = "1"
        return redirect("/student/write")

    return render_template("student_enter.html")



# ---------- 학생 글쓰기 ----------
@app.route("/student/write", methods=["GET", "POST"])
def student_write():
    if "code" not in session or "name" not in session:
        return redirect("/student")

    code = (session.get("code") or "").upper().strip()
    name = (session.get("name") or "").strip()

    # ✅ sid 먼저 확정 (DB 조회/저장에 필요)
    sid = (session.get("sid") or "1").strip()
    if sid not in ["1", "2", "3", "4", "5"]:
        sid = "1"
        session["sid"] = sid

    # -----------------------------------------
    # ✅ DB 우선
    # -----------------------------------------
    if engine:
        # 1) 학급 존재 + teacher_username 확보
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT code, name, teacher_username
                FROM classes
                WHERE code = :code
                LIMIT 1
            """), {"code": code}).fetchone()

        if not row:
            return redirect("/student")

        # 2) 학생 목록 (친구 목록 구성)
        students = db_get_students_in_class(code)
        all_names = [s["name"] for s in students]

        if name not in all_names:
            return redirect("/student")

        friends = [n for n in all_names if n != name]

        # 3) 화면용 cls 구성 (템플릿 호환: sessions/meta 필요)
        cls = ensure_class_schema({
            "code": code,
            "name": row.name,
            "teacher": row.teacher_username,
            "sessions": {}
        })
        cls["students"] = students

        # ✅ 중요: 템플릿/기존 로직 호환을 위해 students_data 최소 구성
        cls["students_data"] = {n: {"sessions": {}} for n in all_names}

        # 4) DB에서 해당 학생/회차 세션 복원
        db_sess = db_get_student_session(code, name, sid)
        placements = (db_sess.get("placements") if db_sess else {}) or {}
        submitted = bool(db_sess.get("submitted")) if db_sess else False

        # 5) 제출 완료면 GET에서도 바로 submitted 페이지로 보내고 싶으면 여기서 처리 가능(선택)
        # if submitted:
        #     return redirect("/student/submitted")

        # -----------------------------------------
        # POST 처리
        # -----------------------------------------
        if request.method == "POST":
            # 제출 완료면 막기
            if submitted:
                return redirect("/student/submitted")

            placements_json = (request.form.get("placements_json") or "{}").strip()
            try:
                placements_obj = json.loads(placements_json) if placements_json else {}
            except Exception:
                placements_obj = {}

            # 구글 시트 저장(기존 유지)
            resp = post_to_sheet({
                "action": "result_append",
                "teacher": row.teacher_username,
                "class_code": code,
                "student": name,
                "session": sid,
                "placements": placements_obj,
                "ip": request.headers.get("X-Forwarded-For", request.remote_addr) or ""
            })

            if resp.get("status") != "ok":
                return render_template(
                    "student_write.html",
                    error=f"저장 실패(구글 시트): {resp}",
                    name=name,
                    friends=friends,
                    placements=placements_obj,
                    student_session={"placements": placements_obj, "submitted": False},
                    sid=sid,
                    session_meta=cls.get("sessions", {}).get(sid, {}),
                )

            # ✅ DB 저장 (최종)
            db_upsert_student_session(
                class_code=code,
                student_name=name,
                sid=sid,
                placements=placements_obj,
                submitted=True
            )

            return redirect("/student/submitted")

        # -----------------------------------------
        # GET 렌더
        # -----------------------------------------
        return render_template(
            "student_write.html",
            name=name,
            friends=friends,
            placements=placements,
            student_session={"placements": placements, "submitted": submitted},
            sid=sid,
            session_meta=cls.get("sessions", {}).get(sid, {}),
        )

    # -----------------------------------------
    # (예외) DB가 없을 때만 JSON fallback
    # -----------------------------------------
    d = load_data()
    cls = ensure_class_schema(d.get("classes", {}).get(code))
    if not cls:
        return redirect("/student")

    if name not in (cls.get("students_data") or {}):
        return redirect("/student")

    if sid not in cls.get("sessions", {}):
        sid = "1"
        session["sid"] = sid

    student = cls["students_data"][name]
    student.setdefault("sessions", {})
    student["sessions"].setdefault(sid, {"placements": {}, "submitted": False})
    ssession = student["sessions"][sid]

    friends = [s["name"] for s in cls.get("students", []) if isinstance(s, dict) and s.get("name") != name]
    placements = (ssession.get("placements") or {})

    if request.method == "POST":
        if ssession.get("submitted"):
            return redirect("/student/submitted")

        placements_json = (request.form.get("placements_json") or "{}").strip()
        try:
            placements_obj = json.loads(placements_json) if placements_json else {}
        except Exception:
            placements_obj = {}

        resp = post_to_sheet({
            "action": "result_append",
            "teacher": cls.get("teacher", ""),
            "class_code": code,
            "student": name,
            "session": sid,
            "placements": placements_obj,
            "ip": request.headers.get("X-Forwarded-For", request.remote_addr) or ""
        })

        if resp.get("status") != "ok":
            return render_template(
                "student_write.html",
                error=f"저장 실패(구글 시트): {resp}",
                name=name,
                friends=friends,
                placements=placements_obj,
                student_session=ssession,
                sid=sid,
                session_meta=cls.get("sessions", {}).get(sid, {}),
            )

        ssession["placements"] = placements_obj
        ssession["submitted"] = True
        student["sessions"][sid] = ssession
        d["classes"][code] = ensure_class_schema(cls)
        save_data_safely(d)

        return redirect("/student/submitted")

    return render_template(
        "student_write.html",
        name=name,
        friends=friends,
        placements=placements,
        student_session=ssession,
        sid=sid,
        session_meta=cls.get("sessions", {}).get(sid, {}),
    )

import math
from datetime import datetime

def cache_get(class_code: str, sid: str, key: str):
    if not engine:
        return None
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT payload
            FROM analysis_cache
            WHERE class_code=:c AND session_id=:s AND cache_key=:k
            LIMIT 1
        """), {"c": class_code, "s": sid, "k": key}).fetchone()
    return row.payload if row else None

def cache_set(class_code: str, sid: str, key: str, payload: dict):
    if not engine:
        return
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO analysis_cache (class_code, session_id, cache_key, payload, updated_at)
            VALUES (:c, :s, :k, :p, NOW())
            ON CONFLICT (class_code, session_id, cache_key)
            DO UPDATE SET payload = EXCLUDED.payload, updated_at = NOW()
        """), {"c": class_code, "s": sid, "k": key, "p": json.dumps(payload, ensure_ascii=False)})

def _extract_point(v, canvas_w=None, canvas_h=None):
    """
    placements의 한 항목 v에서 2D point를 뽑는다.
    - 신규(abs): {x,y,w,h,mode:'abs'} => (x/w, y/h)로 정규화
    - 구버전(rel): {x,y,d,...} => (x,y) 그대로 사용 (스케일은 후처리에서 자동 정규화)
    """
    if not isinstance(v, dict):
        return None

    mode = v.get("mode")
    x = v.get("x")
    y = v.get("y")
    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
        return None

    if mode == "abs":
        w = v.get("w") if isinstance(v.get("w"), (int, float)) else canvas_w
        h = v.get("h") if isinstance(v.get("h"), (int, float)) else canvas_h
        if not (isinstance(w, (int, float)) and w > 0 and isinstance(h, (int, float)) and h > 0):
            # w/h가 없다면 abs를 그대로 쓰되, 이후 정규화에서 처리
            return (float(x), float(y), "abs_raw")
        return (float(x) / float(w), float(y) / float(h), "abs_norm")

    # rel(구버전): 중심 기준 상대좌표
    return (float(x), float(y), "rel")

def points_from_placements_all_students(placements: dict, names: list[str]):
    """
    교사 배치: 모든 학생이 placements에 있어야 함(프론트에서 강제).
    그래도 안전하게 None 체크.
    반환: points(list[(x,y)]), valid_mask(list[bool])
    """
    pts = []
    valid = []
    for nm in names:
        p = _extract_point(placements.get(nm))
        if p is None:
            pts.append((0.0, 0.0))
            valid.append(False)
        else:
            pts.append((p[0], p[1]))
            valid.append(True)

    # abs_raw/rel 스케일 차이를 줄이기 위해, 전체를 한 번 정규화:
    # - bbox 기준으로 0~1 스케일로 맞춰줌(방향은 의미 없고 거리만 쓰므로 안전)
    xs = [pts[i][0] for i in range(len(pts)) if valid[i]]
    ys = [pts[i][1] for i in range(len(pts)) if valid[i]]
    if len(xs) >= 2 and len(ys) >= 2:
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        rx = (maxx - minx) if (maxx - minx) > 1e-9 else 1.0
        ry = (maxy - miny) if (maxy - miny) > 1e-9 else 1.0
        pts = [((x - minx)/rx, (y - miny)/ry) for (x, y) in pts]
    return pts, valid

def distance_matrix(points: list[tuple[float,float]], valid: list[bool]):
    """
    NxN 거리행렬. 유효하지 않은 노드가 있으면 해당 row/col은 None 처리.
    반환: matrix(list[list[float|None]])
    """
    n = len(points)
    D = [[None]*n for _ in range(n)]
    for i in range(n):
        if not valid[i]: 
            continue
        D[i][i] = 0.0
        xi, yi = points[i]
        for j in range(i+1, n):
            if not valid[j]:
                continue
            xj, yj = points[j]
            d = math.hypot(xi-xj, yi-yj)
            d = round(d, 6)
            D[i][j] = d
            D[j][i] = d
    return D

def mean_distance_matrix(mats: list[list[list[float|None]]]):
    """
    여러 거리행렬의 pairwise mean(결측(None)은 제외).
    반환: avgD
    """
    if not mats:
        return []
    n = len(mats[0])
    avg = [[None]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            vals = []
            for M in mats:
                v = M[i][j]
                if isinstance(v, (int, float)):
                    vals.append(float(v))
            if vals:
                avg[i][j] = round(sum(vals)/len(vals), 6)
    # 대각 0 보정
    for i in range(n):
        avg[i][i] = 0.0
    return avg

def classical_mds_2d(D: list[list[float|None]]):
    """
    Classical MDS(2D). 결측이 있으면 평균으로 대체해 단순히 수행(교사 전원배치에서는 결측 없어야 정상).
    반환: coords(list[(x,y)])
    """
    n = len(D)
    if n == 0:
        return []

    # 1) 결측 대체(가장 안전한 최소 구현): row 평균
    filled = [[0.0]*n for _ in range(n)]
    for i in range(n):
        row_vals = [D[i][j] for j in range(n) if isinstance(D[i][j], (int,float)) and i != j]
        row_mean = (sum(row_vals)/len(row_vals)) if row_vals else 0.0
        for j in range(n):
            v = D[i][j]
            filled[i][j] = float(v) if isinstance(v, (int,float)) else float(row_mean)

    # 2) double-centering: B = -0.5 * J * (D^2) * J
    #    (numpy 없이 순수 파이썬으로 구현)
    D2 = [[filled[i][j]**2 for j in range(n)] for i in range(n)]

    row_mean = [sum(D2[i])/n for i in range(n)]
    col_mean = [sum(D2[i][j] for i in range(n))/n for j in range(n)]
    total_mean = sum(row_mean)/n

    B = [[-0.5 * (D2[i][j] - row_mean[i] - col_mean[j] + total_mean) for j in range(n)] for i in range(n)]

    # 3) eigen decomposition (간단히 power iteration 2회로 2축 근사)
    #    안정성을 위해 매우 작은 구현: 2개 축만 구함
    def matvec(M, v):
        return [sum(M[i][k]*v[k] for k in range(n)) for i in range(n)]

    def dot(a,b): 
        return sum(a[i]*b[i] for i in range(n))

    def norm(v):
        return math.sqrt(dot(v,v)) + 1e-12

    def power_iter(M, iters=80):
        v = [1.0/math.sqrt(n)]*n
        for _ in range(iters):
            w = matvec(M, v)
            nv = norm(w)
            v = [x/nv for x in w]
        lam = dot(v, matvec(M, v))
        return lam, v

    # 첫 축
    lam1, v1 = power_iter(B)
    # deflation
    B2 = [[B[i][j] - lam1*v1[i]*v1[j] for j in range(n)] for i in range(n)]
    # 둘째 축
    lam2, v2 = power_iter(B2)

    lam1 = max(lam1, 0.0)
    lam2 = max(lam2, 0.0)

    s1 = math.sqrt(lam1)
    s2 = math.sqrt(lam2)

    coords = [(round(v1[i]*s1, 6), round(v2[i]*s2, 6)) for i in range(n)]
    return coords

def teacher_run_distance_payload(class_code: str, sid: str, run_id: int):
    """
    teacher_placement_runs(run_id)의 placements로:
    - points(정규화)
    - 거리행렬
    - (선택) MDS 좌표
    를 산출하여 payload 반환
    """
    run = db_get_teacher_run(run_id)
    if not run:
        return None
    placements = run["placements"] or {}

    students = db_get_students_in_class(class_code)
    names = [s["name"] for s in students]

    pts, valid = points_from_placements_all_students(placements, names)
    D = distance_matrix(pts, valid)
    X = classical_mds_2d(D)

    return {
        "class_code": class_code,
        "session_id": sid,
        "run_id": run_id,
        "names": names,
        "points_norm": [{"x": pts[i][0], "y": pts[i][1], "valid": bool(valid[i])} for i in range(len(names))],
        "distance_matrix": D,
        "mds_2d": [{"x": X[i][0], "y": X[i][1]} for i in range(len(names))],
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }



# ---------- 제출 완료 ----------
@app.route("/student/submitted")
def student_submitted():
    return render_template("student_submitted.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
