"""
BizNova backend — Flask app that proxies AI requests to OpenAI
and serves the static HTML frontend.

Run:
    pip install -r requirements.txt
    copy .env.example .env   # then edit and set OPENAI_API_KEY
    python app.py
"""
import os
import re
import sqlite3
import secrets
from datetime import datetime, timedelta, timezone

UTC = timezone.utc
from functools import wraps
from io import BytesIO

from flask import Flask, request, jsonify, send_from_directory, send_file, session
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
from openai import OpenAI, OpenAIError
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, HRFlowable
)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "").strip() or secrets.token_hex(32)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.getenv("BIZNOVA_DB_PATH", os.path.join(BASE_DIR, "biznova.db"))
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
USE_POSTGRES = DATABASE_URL.startswith(("postgres://", "postgresql://"))
IS_PROD = os.getenv("FLASK_ENV", "development").lower() == "production" or os.getenv("RENDER") is not None

app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")
app.secret_key = SECRET_KEY
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=IS_PROD,
    PERMANENT_SESSION_LIFETIME=timedelta(days=14),
)
CORS(app, supports_credentials=True)

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ---- Database --------------------------------------------------------------
# Supports SQLite (local dev) and Postgres (production via DATABASE_URL).
# We write SQL using "?" placeholders and translate to "%s" when on Postgres.

if USE_POSTGRES:
    import psycopg
    from psycopg.rows import dict_row

    # Normalise scheme for psycopg
    _PG_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)


class _Row(dict):
    """Dict that also supports row['key'] AND row[index] for sqlite compat."""
    pass


class _Cursor:
    def __init__(self, raw, is_pg):
        self._raw = raw
        self._is_pg = is_pg

    def execute(self, sql, params=()):
        if self._is_pg:
            sql = sql.replace("?", "%s")
        self._raw.execute(sql, params)
        return self

    def fetchone(self):
        return self._raw.fetchone()

    def fetchall(self):
        return self._raw.fetchall()

    @property
    def rowcount(self):
        return self._raw.rowcount

    @property
    def lastrowid(self):
        if self._is_pg:
            row = self._raw.fetchone()
            return row["id"] if row else None
        return self._raw.lastrowid


class _Conn:
    def __init__(self):
        self._is_pg = USE_POSTGRES
        if self._is_pg:
            self._raw = psycopg.connect(_PG_URL, row_factory=dict_row, autocommit=False)
        else:
            self._raw = sqlite3.connect(DB_PATH)
            self._raw.row_factory = sqlite3.Row

    def execute(self, sql, params=()):
        if self._is_pg:
            # If insert and we want lastrowid, append RETURNING id
            cur = self._raw.cursor()
            sql_pg = sql.replace("?", "%s")
            if sql_pg.lstrip().upper().startswith("INSERT") and "RETURNING" not in sql_pg.upper():
                sql_pg = sql_pg.rstrip().rstrip(";") + " RETURNING id"
            cur.execute(sql_pg, params)
            return _Cursor(cur, True)
        cur = self._raw.execute(sql, params)
        return _Cursor(cur, False)

    def executescript(self, sql):
        if self._is_pg:
            cur = self._raw.cursor()
            cur.execute(sql)
        else:
            self._raw.executescript(sql)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self._raw.commit()
        else:
            self._raw.rollback()
        self._raw.close()
        return False


def get_db():
    return _Conn()


def init_db():
    if USE_POSTGRES:
        schema = """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            first_name TEXT,
            last_name TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS plans (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_plans_user ON plans(user_id, updated_at DESC);
        CREATE TABLE IF NOT EXISTS logos (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            style TEXT,
            prompt TEXT,
            image_b64 TEXT NOT NULL,
            mime TEXT NOT NULL DEFAULT 'image/png',
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_logos_user ON logos(user_id, created_at DESC);
        """
    else:
        schema = """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            first_name TEXT,
            last_name TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_plans_user ON plans(user_id, updated_at DESC);
        CREATE TABLE IF NOT EXISTS logos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            style TEXT,
            prompt TEXT,
            image_b64 TEXT NOT NULL,
            mime TEXT NOT NULL DEFAULT 'image/png',
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_logos_user ON logos(user_id, created_at DESC);
        DROP TABLE IF EXISTS otps;
        """
    with get_db() as conn:
        conn.executescript(schema)


# ---- Auth helpers ----------------------------------------------------------

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def is_valid_email(s: str) -> bool:
    return bool(s) and bool(EMAIL_RE.match(s))


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            return jsonify({"error": "Authentication required"}), 401
        return fn(*args, **kwargs)
    return wrapper


def current_user():
    uid = session.get("user_id")
    if not uid:
        return None
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, email, first_name, last_name FROM users WHERE id=?",
            (uid,),
        ).fetchone()
    return dict(row) if row else None


# ---- Auth routes -----------------------------------------------------------

@app.route("/api/auth/me")
def auth_me():
    user = current_user()
    if not user:
        return jsonify({"authenticated": False}), 200
    return jsonify({"authenticated": True, "user": user})


@app.route("/api/auth/register", methods=["POST"])
def auth_register():
    """Create a new account with email + password."""
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    first = (body.get("first_name") or "").strip()
    last = (body.get("last_name") or "").strip()
    pwd = body.get("password") or ""
    pwd2 = body.get("password_confirm") or ""

    if not is_valid_email(email):
        return jsonify({"error": "Please enter a valid email address."}), 400
    if not first or not last:
        return jsonify({"error": "First and last name are required."}), 400
    if len(pwd) < 8:
        return jsonify({"error": "Password must be at least 8 characters."}), 400
    if pwd != pwd2:
        return jsonify({"error": "Passwords do not match."}), 400

    pwd_hash = generate_password_hash(pwd)
    try:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO users(email, password_hash, first_name, last_name, created_at) VALUES(?,?,?,?,?)",
                (email, pwd_hash, first, last, datetime.now(UTC).isoformat()),
            )
            row = conn.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()
    except sqlite3.IntegrityError:
        return jsonify({"error": "An account with this email already exists. Please sign in instead."}), 409
    except Exception as e:
        if "duplicate key" in str(e).lower() or "unique constraint" in str(e).lower():
            return jsonify({"error": "An account with this email already exists. Please sign in instead."}), 409
        raise

    session.permanent = True
    session["user_id"] = row["id"]
    return jsonify({"ok": True, "user": current_user()})


@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    """Sign in with email + password."""
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    pwd = body.get("password") or ""
    if not is_valid_email(email) or not pwd:
        return jsonify({"error": "Email and password are required."}), 400

    with get_db() as conn:
        row = conn.execute(
            "SELECT id, password_hash FROM users WHERE email=?", (email,)
        ).fetchone()
    if not row or not row["password_hash"] or not check_password_hash(row["password_hash"], pwd):
        return jsonify({"error": "Incorrect email or password."}), 401

    session.permanent = True
    session["user_id"] = row["id"]
    return jsonify({"ok": True, "user": current_user()})


@app.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    session.clear()
    return jsonify({"ok": True})



SYSTEM_PROMPTS = {
    "general": (
        "You are Nova, an expert South African business advisor on the BizNova platform. "
        "You help aspiring entrepreneurs start and grow businesses in South Africa. "
        "You provide practical, plain-language advice on business planning, CIPC registration, "
        "funding, taxes (SARS), compliance, and general business strategy. Always tailor advice "
        "to the South African context (mention relevant institutions like CIPC, SARS, SEDA, NEF, "
        "IDC, etc.). Be warm, encouraging, and clear. Keep responses concise but thorough. "
        "Where relevant, remind users that for legal or financial matters they should consult a "
        "qualified professional."
    ),
    "business-plan": (
        "You are Nova, a business plan specialist on BizNova. Help users write, refine, and "
        "improve business plans for South African businesses. Ask clarifying questions if needed. "
        "Suggest improvements. Explain the purpose of each section. Be specific and practical."
    ),
    "legal": (
        "You are Nova, a South African company registration guide on BizNova. Explain CIPC "
        "registration, company types (Pty Ltd, NPC, CC conversion, etc.), required documents, "
        "fees, and processes in plain language. Always remind users this is general guidance and "
        "not legal advice, and to verify at cipc.co.za or consult a professional."
    ),
    "funding": (
        "You are Nova, a South African business funding advisor on BizNova. Help users understand "
        "loan options (SEDA, IDC, NEF, NYDA, bank loans, microfinance), grant programmes, angel "
        "investors, and government support. Always clarify you're providing general information "
        "and not financial advice."
    ),
    "compliance": (
        "You are Nova, a South African business compliance guide on BizNova. Help users understand "
        "tax registration (SARS), UIF, COIDA, B-BBEE, labour law, and other compliance "
        "requirements. Use plain language and always note that specific advice should come from a "
        "qualified professional."
    ),
}


def call_openai(messages, system_prompt, max_tokens=1500):
    if client is None:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=full_messages,
        max_tokens=max_tokens,
        temperature=0.7,
    )
    return completion.choices[0].message.content or "(No response)"


# ---- Routes -----------------------------------------------------------------

@app.route("/")
def index():
    if not session.get("user_id"):
        return send_from_directory(BASE_DIR, "login.html")
    return send_from_directory(BASE_DIR, "biznova_app.html")


@app.route("/login")
def login_page():
    return send_from_directory(BASE_DIR, "login.html")


@app.route("/api/health")
def health():
    return jsonify({
        "ok": True,
        "model": OPENAI_MODEL,
        "key_configured": bool(OPENAI_API_KEY),
    })


@app.route("/api/chat", methods=["POST"])
@login_required
def chat():
    body = request.get_json(silent=True) or {}
    mode = body.get("mode", "general")
    history = body.get("messages", [])
    if not isinstance(history, list) or not history:
        return jsonify({"error": "messages must be a non-empty list"}), 400

    clean = []
    for m in history:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            clean.append({"role": role, "content": content})
    if not clean:
        return jsonify({"error": "No valid messages provided"}), 400

    system_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["general"])
    try:
        reply = call_openai(clean, system_prompt, max_tokens=1200)
    except OpenAIError as e:
        return jsonify({"error": f"OpenAI error: {e}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"reply": reply})


@app.route("/api/plan", methods=["POST"])
@login_required
def generate_plan():
    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "").strip()
    idea = (body.get("idea") or "").strip()
    if not name or not idea:
        return jsonify({"error": "name and idea are required"}), 400

    fields = {
        "Business Name": name,
        "Industry": body.get("industry") or "Not specified",
        "Business Idea": idea,
        "Location": body.get("location") or "South Africa",
        "Target Customers": body.get("customers") or "Not specified",
        "Problem Solved": body.get("problem") or "Not specified",
        "Startup Cost": body.get("startup") or "Not specified",
        "Expected Monthly Revenue": body.get("revenue") or "Not specified",
        "Funding Strategy": body.get("funding") or "Not specified",
        "Competitive Advantage": body.get("advantage") or "Not specified",
        "Planned Employees": body.get("employees") or "Not specified",
    }
    details = "\n".join(f"{k}: {v}" for k, v in fields.items())
    prompt = f"""Generate a comprehensive, professional business plan for:

{details}

Please write a professional business plan with these sections:
1. Executive Summary
2. Company Overview
3. Products & Services
4. Market Analysis (South African market context where relevant)
5. Marketing & Sales Strategy
6. Operational Plan
7. Management & Team
8. Financial Plan (including startup costs, revenue projections, and break-even analysis)
9. Funding Requirements
10. Risk Analysis & Mitigation

Write in plain English, be specific to the South African business environment where relevant.
Include practical, actionable content. Use clear section headings."""

    try:
        plan = call_openai(
            [{"role": "user", "content": prompt}],
            SYSTEM_PROMPTS["business-plan"],
            max_tokens=3000,
        )
    except OpenAIError as e:
        return jsonify({"error": f"OpenAI error: {e}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Auto-save to user's library
    now = datetime.now(UTC).isoformat()
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO plans(user_id, name, content, created_at, updated_at) VALUES(?,?,?,?,?)",
            (session["user_id"], name, plan, now, now),
        )
        plan_id = cur.lastrowid
    return jsonify({"plan": plan, "name": name, "id": plan_id})


@app.route("/api/plan/download", methods=["POST"])
@login_required
def download_plan():
    """Return the plan as a .doc-compatible HTML file (opens cleanly in Word)."""
    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "BusinessPlan").strip()
    plan = body.get("plan") or ""
    safe_name = "".join(c for c in name if c.isalnum() or c in (" ", "_", "-")).strip().replace(" ", "_") or "BusinessPlan"

    safe_plan = plan.replace("<", "&lt;").replace(">", "&gt;")
    html = f"""<html xmlns:o='urn:schemas-microsoft-com:office:office'
xmlns:w='urn:schemas-microsoft-com:office:word' xmlns='http://www.w3.org/TR/REC-html40'>
<head><meta charset='utf-8'><title>{name} — Business Plan</title>
<style>body{{font-family:Calibri,Arial,sans-serif;font-size:11pt;line-height:1.5;color:#222}}
h1{{font-size:22pt;color:#1A5C3A}}h2{{font-size:14pt;color:#1A5C3A;border-bottom:1px solid #C9A84C;padding-bottom:4px}}
pre{{white-space:pre-wrap;font-family:Calibri,Arial,sans-serif;font-size:11pt}}</style></head>
<body><h1>{name}</h1><h2>Business Plan</h2><pre>{safe_plan}</pre></body></html>"""

    buf = BytesIO(html.encode("utf-8"))
    return send_file(
        buf,
        mimetype="application/msword",
        as_attachment=True,
        download_name=f"{safe_name}_BusinessPlan.doc",
    )


# ---- PDF export -------------------------------------------------------------

GREEN = HexColor("#1A5C3A")
GOLD = HexColor("#C9A84C")
INK = HexColor("#0F1A12")
INK_60 = HexColor("#666666")


def _safe_filename(name: str) -> str:
    cleaned = "".join(c for c in name if c.isalnum() or c in (" ", "_", "-")).strip()
    return cleaned.replace(" ", "_") or "BusinessPlan"


def _inline_md(text: str) -> str:
    """Convert basic Markdown emphasis + escape XML for ReportLab Paragraph."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", text)
    return text


def _build_pdf(business_name: str, plan_text: str) -> BytesIO:
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title=f"{business_name} — Business Plan",
        author="BizNova",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "BNTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=26, leading=32, textColor=GREEN, alignment=TA_LEFT, spaceAfter=4,
    )
    sub_style = ParagraphStyle(
        "BNSub", parent=styles["Normal"], fontName="Helvetica",
        fontSize=11, textColor=INK_60, spaceAfter=18,
    )
    h1_style = ParagraphStyle(
        "BNH1", parent=styles["Heading1"], fontName="Helvetica-Bold",
        fontSize=16, leading=20, textColor=GREEN, spaceBefore=16, spaceAfter=8,
    )
    h2_style = ParagraphStyle(
        "BNH2", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=13, leading=17, textColor=GREEN, spaceBefore=12, spaceAfter=6,
    )
    h3_style = ParagraphStyle(
        "BNH3", parent=styles["Heading3"], fontName="Helvetica-Bold",
        fontSize=11.5, leading=15, textColor=INK, spaceBefore=10, spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "BNBody", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=10.5, leading=15, textColor=INK, spaceAfter=8, alignment=TA_LEFT,
    )
    bullet_style = ParagraphStyle(
        "BNBullet", parent=body_style, leftIndent=18, bulletIndent=6, spaceAfter=4,
    )
    footer_style = ParagraphStyle(
        "BNFooter", parent=styles["Normal"], fontName="Helvetica",
        fontSize=8.5, textColor=INK_60, alignment=TA_CENTER,
    )

    story = []
    story.append(Paragraph(business_name, title_style))
    story.append(Paragraph("Business Plan · Generated by BizNova", sub_style))
    story.append(HRFlowable(width="100%", thickness=1.2, color=GOLD, spaceBefore=0, spaceAfter=14))

    lines = (plan_text or "").splitlines()
    in_list = False

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            in_list = False
            story.append(Spacer(1, 4))
            continue

        # Headings
        if line.startswith("### "):
            in_list = False
            story.append(Paragraph(_inline_md(line[4:].strip()), h3_style))
            continue
        if line.startswith("## "):
            in_list = False
            story.append(Paragraph(_inline_md(line[3:].strip()), h2_style))
            continue
        if line.startswith("# "):
            in_list = False
            story.append(Paragraph(_inline_md(line[2:].strip()), h1_style))
            continue

        # Numbered section heading like "1. Executive Summary"
        m = re.match(r"^(\d{1,2})\.\s+([A-Z][^\n]{2,80})$", line.strip())
        if m and not line.lstrip().startswith("-") and len(m.group(2).split()) <= 8:
            in_list = False
            story.append(Paragraph(_inline_md(line.strip()), h2_style))
            continue

        # Bullet list item
        bullet_match = re.match(r"^\s*[-*•]\s+(.*)$", line)
        if bullet_match:
            in_list = True
            story.append(Paragraph(_inline_md(bullet_match.group(1)), bullet_style, bulletText="•"))
            continue

        in_list = False
        story.append(Paragraph(_inline_md(line.strip()), body_style))

    def _on_page(canvas, _doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8.5)
        canvas.setFillColor(INK_60)
        footer = f"BizNova · {business_name} · Page {_doc.page} · {datetime.now():%d %b %Y}"
        canvas.drawCentredString(A4[0] / 2, 1.2 * cm, footer)
        canvas.setStrokeColor(GOLD)
        canvas.setLineWidth(0.6)
        canvas.line(2 * cm, 1.5 * cm, A4[0] - 2 * cm, 1.5 * cm)
        canvas.restoreState()

    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    buf.seek(0)
    return buf


@app.route("/api/plan/pdf", methods=["POST"])
@login_required
def plan_pdf():
    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "BusinessPlan").strip() or "BusinessPlan"
    plan = body.get("plan") or ""
    if not plan.strip():
        return jsonify({"error": "plan content is required"}), 400
    try:
        pdf = _build_pdf(name, plan)
    except Exception as e:
        return jsonify({"error": f"PDF generation failed: {e}"}), 500
    return send_file(
        pdf,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"{_safe_filename(name)}_BusinessPlan.pdf",
    )


# ---- Logo generation -------------------------------------------------------

LOGO_STYLE_HINTS = {
    "modern": "modern, clean, minimalist, geometric shapes, professional",
    "classic": "classic, elegant, timeless, refined typography, traditional emblem",
    "playful": "playful, friendly, vibrant colours, rounded shapes, approachable",
    "luxury": "luxury, premium, gold and deep tones, sophisticated, high-end",
    "tech": "tech, futuristic, gradient, sleek, app-icon style",
    "earthy": "earthy, organic, natural tones, hand-crafted feel, African-inspired motifs",
}


@app.route("/api/plan/logo", methods=["POST"])
@login_required
def generate_logo():
    if client is None:
        return jsonify({"error": "OPENAI_API_KEY is not set on the server."}), 500
    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "").strip()
    industry = (body.get("industry") or "").strip()
    idea = (body.get("idea") or "").strip()
    style = (body.get("style") or "modern").strip().lower()
    user_prompt = (body.get("prompt") or "").strip()
    if not name:
        return jsonify({"error": "Business name is required."}), 400

    style_hint = LOGO_STYLE_HINTS.get(style, LOGO_STYLE_HINTS["modern"])
    prompt_parts = [
        f'A professional business logo for a South African company called "{name}".',
        f"Style: {style_hint}.",
        f"Industry: {industry or 'general business'}.",
    ]
    if user_prompt:
        prompt_parts.append(f"User direction: {user_prompt}")
    elif idea:
        prompt_parts.append(f"Concept: {idea}")
    prompt_parts.append(
        "The logo should be a clean vector-style mark on a plain white background, "
        "centred, balanced composition, suitable for use on business cards, websites and signage. "
        "Include the company name as part of the logo in clear, legible typography. "
        "No mock-ups, no extra text, no watermarks, no photographs."
    )
    prompt = " ".join(prompt_parts)

    try:
        result = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
            response_format="b64_json",
        )
        b64 = result.data[0].b64_json
    except OpenAIError as e:
        return jsonify({"error": f"Image generation failed: {e}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Auto-save to user's library
    now = datetime.now(UTC).isoformat()
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO logos(user_id, name, style, prompt, image_b64, mime, created_at) VALUES(?,?,?,?,?,?,?)",
            (session["user_id"], name, style, user_prompt or idea, b64, "image/png", now),
        )
        logo_id = cur.lastrowid

    return jsonify({
        "ok": True,
        "id": logo_id,
        "image_b64": b64,
        "mime": "image/png",
        "style": style,
        "filename": f"{_safe_filename(name)}_logo.png",
    })


# ---- Library / Dashboard ---------------------------------------------------

@app.route("/api/plans", methods=["GET"])
@login_required
def list_plans():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, name, substr(content,1,200) AS preview, created_at, updated_at "
            "FROM plans WHERE user_id=? ORDER BY updated_at DESC",
            (session["user_id"],),
        ).fetchall()
    return jsonify({"plans": [dict(r) for r in rows]})


@app.route("/api/plans/<int:plan_id>", methods=["GET"])
@login_required
def get_plan(plan_id):
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, name, content, created_at, updated_at FROM plans WHERE id=? AND user_id=?",
            (plan_id, session["user_id"]),
        ).fetchone()
    if not row:
        return jsonify({"error": "Plan not found."}), 404
    return jsonify(dict(row))


@app.route("/api/plans/<int:plan_id>", methods=["PUT"])
@login_required
def update_plan(plan_id):
    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "").strip()
    content = body.get("content") or ""
    if not name or not content.strip():
        return jsonify({"error": "Name and content are required."}), 400
    now = datetime.now(UTC).isoformat()
    with get_db() as conn:
        cur = conn.execute(
            "UPDATE plans SET name=?, content=?, updated_at=? WHERE id=? AND user_id=?",
            (name, content, now, plan_id, session["user_id"]),
        )
        if cur.rowcount == 0:
            return jsonify({"error": "Plan not found."}), 404
    return jsonify({"ok": True, "updated_at": now})


@app.route("/api/plans/<int:plan_id>", methods=["DELETE"])
@login_required
def delete_plan(plan_id):
    with get_db() as conn:
        cur = conn.execute(
            "DELETE FROM plans WHERE id=? AND user_id=?",
            (plan_id, session["user_id"]),
        )
        if cur.rowcount == 0:
            return jsonify({"error": "Plan not found."}), 404
    return jsonify({"ok": True})


@app.route("/api/logos", methods=["GET"])
@login_required
def list_logos():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, name, style, prompt, created_at FROM logos WHERE user_id=? ORDER BY created_at DESC",
            (session["user_id"],),
        ).fetchall()
    return jsonify({"logos": [dict(r) for r in rows]})


@app.route("/api/logos/<int:logo_id>", methods=["GET"])
@login_required
def get_logo(logo_id):
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, name, style, prompt, image_b64, mime, created_at FROM logos WHERE id=? AND user_id=?",
            (logo_id, session["user_id"]),
        ).fetchone()
    if not row:
        return jsonify({"error": "Logo not found."}), 404
    return jsonify(dict(row))


@app.route("/api/logos/<int:logo_id>", methods=["DELETE"])
@login_required
def delete_logo(logo_id):
    with get_db() as conn:
        cur = conn.execute(
            "DELETE FROM logos WHERE id=? AND user_id=?",
            (logo_id, session["user_id"]),
        )
        if cur.rowcount == 0:
            return jsonify({"error": "Logo not found."}), 404
    return jsonify({"ok": True})


# Initialise DB at import time so gunicorn/wsgi workers also create tables.
if not USE_POSTGRES:
    try:
        os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    except OSError:
        pass
init_db()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5001"))
    print(f"BizNova backend running at http://localhost:{port}")
    if not OPENAI_API_KEY:
        print("WARNING: OPENAI_API_KEY not set — AI endpoints will return errors.")
    app.run(host="0.0.0.0", port=port, debug=not IS_PROD)
