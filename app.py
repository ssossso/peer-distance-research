from flask import Flask, render_template, request, redirect, session, send_file
import random, string, json, os
from urllib.parse import quote, unquote
import io
import sys
import subprocess
import requests

"""Flask app for class relationship measurement.

Optional dependencies (e.g., qrcode) are imported lazily so the app can still
start even if they are not installed yet.
"""

app = Flask(__name__)
app.secret_key = "secret-key"

SITE_TITLE = "내가 바라본 우리 반"
DATA_FILE = "data.json"

GOOGLE_SHEET_URL = "https://script.google.com/macros/s/AKfycbyPnpeTtY2eva-7yJgi3ql2iquROaitNJmULGGbYyfdQqh_4YnXspu88L9osX3mJaTx/exec"
GOOGLE_SECRET = "my_super_secret_key_2026"

# ---------- 데이터 ----------

### ★ Google Sheets로 저장하는 함수 ★
def save_to_google_sheet(student, session_id, placements, ip):
    payload = {
        "student": student,
        "session": session_id,
        "placements": placements,
        "ip": ip,
        "secret": GOOGLE_SECRET
    }
    try:
        requests.post(GOOGLE_SHEET_URL, json=payload, timeout=5)
    except Exception as e:
        print("Google Sheets 저장 실패:", e)


def load_data():
    if not os.path.exists(DATA_FILE):
        return {"teachers": {}, "classes": {}}
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        d = json.load(f)
    # 하위 호환 보정
    for code, cls in (d.get("classes") or {}).items():
        d["classes"][code] = ensure_class_schema(cls)
    return d

def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def make_code():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

def ensure_class_schema(cls):
    """데이터 스키마를 최신 형태로 보정합니다(하위 호환).

    - 회차(sessions) 구조를 보장
    - 과거 버전(placements/submitted가 학생 루트에 있던 형태)을 1차 회차로 이관
    """
    if not cls:
        return cls

    # 1) 학급 회차 목록(최소 1회, 최대 5회)
    if "sessions" not in cls or not isinstance(cls.get("sessions"), dict):
        cls["sessions"] = {}

    # 1~5차 기본 회차를 항상 만들어 둠(교사는 링크만 배포하면 됨)
    for i in range(1, 6):
        sid = str(i)
        cls["sessions"].setdefault(sid, {"label": f"{i}차", "active": True if i == 1 else False})

    # 2) 학생 데이터 회차 구조
    for name, sdata in (cls.get("students_data") or {}).items():
        if not isinstance(sdata, dict):
            continue

        if "sessions" not in sdata or not isinstance(sdata.get("sessions"), dict):
            sdata["sessions"] = {}

        # 과거 필드 이관(1차로)
        if ("placements" in sdata) or ("submitted" in sdata):
            legacy = {
                "placements": sdata.pop("placements", {}) or {},
                "submitted": bool(sdata.pop("submitted", False)),
            }
            # text 필드가 있었던 과거 버전까지 고려
            if "text" in sdata:
                legacy["text"] = sdata.pop("text", "")
            sdata["sessions"].setdefault("1", {})
            sdata["sessions"]["1"].update(legacy)

        # 각 회차 기본값 보장
        for sid in cls["sessions"].keys():
            sdata["sessions"].setdefault(sid, {"placements": {}, "submitted": False})

    return cls

def save_data_safely(d):
    """저장 전 모든 학급 스키마 보정"""
    for code, cls in (d.get("classes") or {}).items():
        d["classes"][code] = ensure_class_schema(cls)
    save_data(d)

def get_current_class():
    """
    상단바에 표시할 '현재 선택 학급'을 계산합니다.
    - 교사: session['selected_class']가 있으면 그 학급
    - 학생: session['code']가 있으면 그 학급
    """
    d = load_data()
    code = None

    if "teacher" in session and session.get("selected_class"):
        code = session.get("selected_class")
    elif session.get("code"):
        code = session.get("code")

    if not code:
        return None

    cls = d.get("classes", {}).get(code)
    if not cls:
        return None

    return {"name": cls.get("name", ""), "code": code}

@app.context_processor
def inject_globals():
    return {
        "current_class": get_current_class()
    }

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

        d = load_data()
        if username in d["teachers"]:
            return render_template("teacher_signup.html", error="이미 존재하는 아이디입니다.")

        d["teachers"][username] = pw
        save_data(d)
        return redirect("/teacher/login")

    return render_template("teacher_signup.html")

# ---------- 교사 로그인 ----------
@app.route("/teacher/login", methods=["GET", "POST"])
def teacher_login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        pw = request.form.get("password", "")

        d = load_data()
        if d["teachers"].get(username) == pw:
            session.clear()
            session["teacher"] = username
            return redirect("/teacher/dashboard")
        return render_template("teacher_login.html", error="로그인 실패")

    return render_template("teacher_login.html")

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

    d = load_data()
    classes = {c: v for c, v in d["classes"].items() if v["teacher"] == session["teacher"]}

    # 대시보드에서는 '선택 학급'이 없을 수 있으니 그대로 둠(유지되게 하려면 clear하지 않음)
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
        # create_class.html에서 "번호\t이름" 형태로 전달(과거 버전 호환: 이름만 올 수도 있음)
        students_raw = request.form.get("students", "")

        parsed = []  # [{"no": "1", "name": "홍길동"}, ...]
        auto_no = 1
        for line in students_raw.splitlines():
            line = (line or "").strip()
            if not line:
                continue

            # 탭 구분(엑셀/구글시트 복붙), 콤마 구분도 허용
            parts = [p.strip() for p in line.split("\t")]
            if len(parts) == 1:
                parts = [p.strip() for p in line.split(",")]

            if len(parts) >= 2:
                no, name = parts[0], parts[1]
            else:
                no, name = "", parts[0]

            name = (name or "").strip()
            if not name:
                continue

            no = (no or "").strip() or str(auto_no)
            auto_no += 1
            parsed.append({"no": no, "name": name})

        # students_data는 "이름" 키를 유지(학생 로그인 검증) + 분석용 회차 구조는 sessions로 관리
        cls = {
            "name": class_name or f"학급 {code}",
            "teacher": session["teacher"],
            "students": parsed,
            "sessions": {"1": {"label": "1차", "active": True}},
            "students_data": {
                s["name"]: {"sessions": {"1": {"placements": {}, "submitted": False}}}
                for s in parsed
            },
        }
        d["classes"][code] = ensure_class_schema(cls)
        save_data_safely(d)
        return redirect("/teacher/dashboard")

    return render_template("create_class.html")

# ---------- 학급 상세 ----------
@app.route("/teacher/class/<code>")
def class_detail(code):
    if "teacher" not in session:
        return redirect("/teacher/login")

    d = load_data()
    cls = ensure_class_schema(d["classes"].get(code))
    if not cls or cls.get("teacher") != session["teacher"]:
        return "학급을 찾을 수 없거나 접근 권한이 없습니다.", 404

    # 회차 선택(쿼리스트링 우선)
    sid = (request.args.get("sid") or session.get("selected_session") or "1").strip()
    if sid not in cls.get("sessions", {}):
        sid = "1"
    session["selected_session"] = sid

    # 현재 선택 학급을 '유지'하기 위해 세션에 저장
    session["selected_class"] = code

    # 학생 목록 + 상태 (과거 데이터 호환: students가 ["이름", ...]일 수도 있음)
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

    # 회차 링크(학생은 회차 선택 없이 링크로 들어옴)
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


# ---------- QR 코드(회차별 입장 링크 배포용) ----------
@app.route("/qr/<code>/<sid>.png")
def qr_session_link(code, sid):
    """회차별 학생 입장 링크를 QR 코드(PNG)로 제공합니다.

    - 학생이 바로 스캔해서 접속할 수 있도록 '절대 URL'을 QR에 넣습니다.
    - Render 등 클라우드에서는 request.url_root가 서비스 도메인을 포함합니다.
    """
    d = load_data()
    cls = d.get("classes", {}).get(code)
    if not cls:
        return "학급을 찾을 수 없습니다.", 404

    cls = ensure_class_schema(cls)
    sid = str(sid).strip()
    if sid not in cls.get("sessions", {}):
        return "회차를 찾을 수 없습니다.", 404

    # 절대 URL 생성
    base = request.url_root.rstrip("/")
    target = f"{base}/s/{code}/{sid}"

    try:
        import qrcode  # type: ignore
    except ModuleNotFoundError:
        # 로컬에서 사용자가 'python app.py'로 바로 실행해도 QR이 자동 생성되도록
        # 가능한 경우 requirements.txt를 자동 설치 후 재시도합니다.
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            import qrcode  # type: ignore
        except Exception:
            msg = (
                "QR 코드 이미지를 생성하려면 필요한 라이브러리 설치가 필요합니다.\n"
                "다음을 실행해 주세요: python -m pip install -r requirements.txt"
            )
            return msg, 500

    img = qrcode.make(target)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    resp = send_file(buf, mimetype="image/png")
    # 캐시 허용(링크가 고정이므로)
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp


# ---------- 교사용 학생 개별 페이지 ----------
@app.route("/teacher/class/<code>/student/<path:student_name>")
def teacher_student_detail(code, student_name):
    if "teacher" not in session:
        return redirect("/teacher/login")

    d = load_data()
    cls = d["classes"].get(code)
    if not cls or cls.get("teacher") != session["teacher"]:
        return "학급을 찾을 수 없거나 접근 권한이 없습니다.", 404

    # URL 인코딩된 학생 이름 복원
    name = unquote(student_name).strip()
    if name not in cls.get("students_data", {}):
        return "학생을 찾을 수 없습니다.", 404

    cls = ensure_class_schema(cls)

    # 회차 선택(기본: 학급 상세에서 선택된 회차 또는 1차)
    sid = (request.args.get("sid") or session.get("selected_session") or "1").strip()
    if sid not in cls.get("sessions", {}):
        sid = "1"

    student = cls["students_data"][name]
    ssession = (student.get("sessions") or {}).get(sid, {"placements": {}, "submitted": False})
    status = "완료" if ssession.get("submitted") else "미완료"

    # 번호 찾기(없으면 빈 값)
    no = ""
    for i, item in enumerate(cls.get("students", []), start=1):
        if isinstance(item, dict) and (item.get("name") or "").strip() == name:
            no = str(item.get("no", "") or i)
            break
        if not isinstance(item, dict) and (item or "").strip() == name:
            no = str(i)
            break

    # 상단바 '현재 선택 학급' 유지
    session["selected_class"] = code

    return render_template(
        "teacher_student_detail.html",
        cls=cls,
        code=code,
        name=name,
        no=no,
        status=status,
        student=student,
        sid=sid,
        session_meta=cls.get("sessions", {}).get(sid, {}),
        session_data=ssession,
    )

# ---------- 학생 입장 ----------
@app.route("/s/<code>/<sid>", methods=["GET", "POST"])
def student_enter_session(code, sid):
    """회차 선택 없이 링크로 입장(교사가 회차별 링크 배포)."""
    code = (code or "").upper().strip()
    sid = (sid or "1").strip()

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
        # 상단바용
        session["selected_class"] = code
        session["selected_session"] = sid
        return redirect("/student/write")

    return render_template(
        "student_enter_session.html",
        code=code,
        sid=sid,
        session_label=cls.get("sessions", {}).get(sid, {}).get("label", f"{sid}차"),
    )

@app.route("/student", methods=["GET", "POST"])
def student_enter():
    if request.method == "POST":
        code = request.form.get("code", "").upper().strip()
        name = request.form.get("name", "").strip()

        d = load_data()
        cls = d["classes"].get(code)
        if not cls:
            return render_template("student_enter.html", error="학급 코드가 올바르지 않습니다.")
        if name not in cls.get("students_data", {}):
            return render_template("student_enter.html", error="학생 명단에 없는 이름입니다.")

        session["code"] = code
        session["name"] = name
        session["sid"] = session.get("sid") or "1"
        return redirect("/student/write")

    return render_template("student_enter.html")


# ---------- 학생 글쓰기 ----------
@app.route("/student/write", methods=["GET", "POST"])
def student_write():
    if "code" not in session or "name" not in session:
        return redirect("/student")

    d = load_data()
    code = session["code"]
    name = session["name"]

    cls = ensure_class_schema(d["classes"].get(code))
    if not cls or name not in cls.get("students_data", {}):
        return redirect("/student")

    sid = (session.get("sid") or "1").strip()
    if sid not in cls.get("sessions", {}):
        sid = "1"
        session["sid"] = sid

    student = cls["students_data"][name]
    ssession = (student.get("sessions") or {}).get(sid, {"placements": {}, "submitted": False})

    # 친구 목록(본인 제외). 과거 데이터 호환: students가 ["이름", ...]일 수도 있음
    friends = []
    for item in cls.get("students", []):
        if isinstance(item, dict):
            n = (item.get("name") or "").strip()
        else:
            n = (item or "").strip()

        if not n or n == name:
            continue
        friends.append(n)

    # 기존 배치(임시저장/제출완료 모두) 로드
    placements = ssession.get("placements", {}) or {}

    if request.method == "POST":
        if ssession.get("submitted"):
            return redirect("/student/submitted")

        placements_json = request.form.get("placements_json", "").strip()
        try:
            placements_obj = json.loads(placements_json) if placements_json else {}
        except Exception:
            placements_obj = {}

        # 저장 포맷: {"친구이름": {"x": int, "y": int, "d": float}}
        ssession["placements"] = placements_obj
        ssession["submitted"] = True
        student.setdefault("sessions", {})[sid] = ssession
        d["classes"][code] = ensure_class_schema(cls)
        save_data_safely(d)

        ### ★ Google Sheets에도 저장 ★
        save_to_google_sheet(
            student=name,
            session_id=sid,
            placements=placements_obj,
            ip=request.remote_addr
        )
        return redirect("/student/submitted")

    return render_template(
        "student_write.html",
        name=name,
        student_session=ssession,
        sid=sid,
        session_meta=cls.get("sessions", {}).get(sid, {}),
        friends=friends,
        placements=placements,
    )

# ---------- 제출 완료 ----------
@app.route("/student/submitted")
def student_submitted():
    if "code" not in session or "name" not in session:
        return redirect("/student")
    return render_template("student_submitted.html")

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
