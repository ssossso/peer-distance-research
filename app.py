import requests
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, render_template, request, redirect, session, send_file
import random, string, json, os
from urllib.parse import quote, unquote
import io
import sys
import subprocess

app = Flask(__name__)
app.secret_key = "secret-key"

SITE_TITLE = "내가 바라본 우리 반"
DATA_FILE = "data.json"

# --- Google Sheets 연동 ---
GOOGLE_WEBAPP_URL = "https://script.google.com/macros/s/AKfycbwyjKC2JearJnySkxdG0oahMkMJ5V6uBqY5EYRGVVRa8KWZvRzHcskeVNY5hnlyiSw/exec"
GOOGLE_SECRET = os.environ.get("GOOGLE_SECRET", "").strip()

# ---------- Google Sheets POST ----------
def post_to_sheet(payload: dict) -> dict:
    payload = dict(payload)
    payload["secret"] = GOOGLE_SECRET
    r = requests.post(GOOGLE_WEBAPP_URL, json=payload, timeout=10)
    return r.json()

# ---------- 데이터 ----------
def load_data():
    if not os.path.exists(DATA_FILE):
        return {"classes": {}}
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        d = json.load(f)
    for code, cls in d.get("classes", {}).items():
        d["classes"][code] = ensure_class_schema(cls)
    return d

def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

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
        except Exception:
            return render_template("teacher_login.html", error="서버 통신 오류")

        if resp.get("status") != "ok":
            return render_template("teacher_login.html", error="로그인 실패")

        if check_password_hash(resp.get("pw_hash", ""), pw):
            session.clear()
            session["teacher"] = username
            return redirect("/teacher/dashboard")

        return render_template("teacher_login.html", error="로그인 실패")

    return render_template("teacher_login.html")

# ---------- 임시디버그 ----------
import hashlib

@app.route("/debug/secret")
def debug_secret():
    s = GOOGLE_SECRET or ""
    return {
        "len": len(s),
        "sha256_8": hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]
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

    d = load_data()
    classes = {c: v for c, v in d["classes"].items() if v["teacher"] == session["teacher"]}
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

        cls = {
            "name": class_name or f"학급 {code}",
            "teacher": session["teacher"],
            "students": parsed,
            "students_data": {
                s["name"]: {"sessions": {}} for s in parsed
            },
        }
        d.setdefault("classes", {})[code] = ensure_class_schema(cls)
        save_data_safely(d)
        return redirect("/teacher/dashboard")

    return render_template("create_class.html")

# ---------- 학생 입장 ----------
@app.route("/student", methods=["GET", "POST"])
def student_enter():
    if request.method == "POST":
        code = request.form.get("code", "").upper().strip()
        name = request.form.get("name", "").strip()

        d = load_data()
        cls = d.get("classes", {}).get(code)
        if not cls or name not in cls["students_data"]:
            return render_template("student_enter.html", error="입장 실패")

        session["code"] = code
        session["name"] = name
        session["sid"] = "1"
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
    cls = d["classes"][code]
    sid = session.get("sid", "1")

    student = cls["students_data"][name]
    ssession = student["sessions"][sid]

    friends = [s["name"] for s in cls["students"] if s["name"] != name]
    placements = ssession.get("placements", {})

    if request.method == "POST":
        placements_obj = json.loads(request.form.get("placements_json", "{}"))

        try:
            post_to_sheet({
                "action": "result_append",
                "teacher": cls["teacher"],
                "class_code": code,
                "student": name,
                "session": sid,
                "placements": placements_obj,
                "ip": request.remote_addr
            })
        except Exception:
            return render_template("student_write.html", error="저장 실패")

        ssession["placements"] = placements_obj
        ssession["submitted"] = True
        save_data_safely(d)
        return redirect("/student/submitted")

    return render_template(
        "student_write.html",
        name=name,
        friends=friends,
        placements=placements,
        student_session=ssession,
        sid=sid
    )

# ---------- 제출 완료 ----------
@app.route("/student/submitted")
def student_submitted():
    return render_template("student_submitted.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
