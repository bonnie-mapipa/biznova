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
import smtplib
from email.message import EmailMessage
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

# SMTP (optional). If not configured, OTPs are printed to console.
SMTP_HOST = os.getenv("SMTP_HOST", "").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "").strip()
SMTP_PASS = os.getenv("SMTP_PASS", "").strip()
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER).strip()
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").lower() != "false"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.getenv("BIZNOVA_DB_PATH", os.path.join(BASE_DIR, "biznova.db"))
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

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT,
            first_name TEXT,
            last_name TEXT,
            email_verified INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS otps (
            email TEXT PRIMARY KEY,
            otp_hash TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            attempts INTEGER DEFAULT 0
        );
        """)


# ---- Auth helpers ----------------------------------------------------------

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
OTP_TTL_MINUTES = 10
OTP_MAX_ATTEMPTS = 5


def is_valid_email(s: str) -> bool:
    return bool(s) and bool(EMAIL_RE.match(s))


def send_otp_email(to_email: str, otp: str) -> None:
    body = (
        f"Your BizNova one-time pin is: {otp}\n\n"
        f"It expires in {OTP_TTL_MINUTES} minutes.\n"
        "If you did not request this, you can ignore this email."
    )
    if not (SMTP_HOST and SMTP_FROM):
        # Dev mode: print to console
        print(f"[OTP] {to_email} -> {otp}  (configure SMTP_* env vars to email instead)")
        return
    msg = EmailMessage()
    msg["Subject"] = "Your BizNova verification code"
    msg["From"] = SMTP_FROM
    msg["To"] = to_email
    msg.set_content(body)
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        if SMTP_USE_TLS:
            s.starttls()
        if SMTP_USER and SMTP_PASS:
            s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)


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


@app.route("/api/auth/send-otp", methods=["POST"])
def auth_send_otp():
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    if not is_valid_email(email):
        return jsonify({"error": "Please enter a valid email address."}), 400

    otp = f"{secrets.randbelow(1000000):06d}"
    otp_hash = generate_password_hash(otp)
    expires = (datetime.now(UTC) + timedelta(minutes=OTP_TTL_MINUTES)).isoformat()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO otps(email, otp_hash, expires_at, attempts) VALUES(?,?,?,0) "
            "ON CONFLICT(email) DO UPDATE SET otp_hash=excluded.otp_hash, "
            "expires_at=excluded.expires_at, attempts=0",
            (email, otp_hash, expires),
        )
        existing = conn.execute(
            "SELECT id, email_verified FROM users WHERE email=?", (email,)
        ).fetchone()

    try:
        send_otp_email(email, otp)
    except Exception as e:
        return jsonify({"error": f"Failed to send email: {e}"}), 500

    return jsonify({
        "ok": True,
        "is_new_user": existing is None or not existing["email_verified"],
        "expires_in_minutes": OTP_TTL_MINUTES,
        # Only echo the OTP back if SMTP is not configured (dev convenience).
        "dev_otp": otp if not SMTP_HOST else None,
    })


@app.route("/api/auth/verify-otp", methods=["POST"])
def auth_verify_otp():
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    otp = (body.get("otp") or "").strip()
    if not is_valid_email(email) or not otp:
        return jsonify({"error": "Email and OTP are required."}), 400

    with get_db() as conn:
        row = conn.execute(
            "SELECT otp_hash, expires_at, attempts FROM otps WHERE email=?", (email,)
        ).fetchone()
        if not row:
            return jsonify({"error": "No OTP requested for this email."}), 400
        if datetime.fromisoformat(row["expires_at"]) < datetime.now(UTC):
            conn.execute("DELETE FROM otps WHERE email=?", (email,))
            return jsonify({"error": "OTP expired. Please request a new one."}), 400
        if row["attempts"] >= OTP_MAX_ATTEMPTS:
            conn.execute("DELETE FROM otps WHERE email=?", (email,))
            return jsonify({"error": "Too many incorrect attempts. Request a new OTP."}), 429
        if not check_password_hash(row["otp_hash"], otp):
            conn.execute(
                "UPDATE otps SET attempts=attempts+1 WHERE email=?", (email,)
            )
            return jsonify({"error": "Incorrect OTP."}), 400

        # OTP valid
        conn.execute("DELETE FROM otps WHERE email=?", (email,))
        existing = conn.execute(
            "SELECT id, email_verified, first_name, password_hash FROM users WHERE email=?",
            (email,),
        ).fetchone()
        if existing is None:
            conn.execute(
                "INSERT INTO users(email, email_verified, created_at) VALUES(?,1,?)",
                (email, datetime.now(UTC).isoformat()),
            )
            is_new = True
        else:
            conn.execute("UPDATE users SET email_verified=1 WHERE email=?", (email,))
            is_new = not (existing["first_name"] and existing["password_hash"])

    # Mark email as verified in session, but do NOT log them in until they
    # complete registration / enter their password.
    session["pending_email"] = email
    return jsonify({"ok": True, "is_new_user": is_new, "email": email})


@app.route("/api/auth/register", methods=["POST"])
def auth_register():
    """Complete account setup after OTP verification (new users)."""
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    if email != session.get("pending_email"):
        return jsonify({"error": "Email not verified in this session."}), 400

    first = (body.get("first_name") or "").strip()
    last = (body.get("last_name") or "").strip()
    pwd = body.get("password") or ""
    pwd2 = body.get("password_confirm") or ""

    if not first or not last:
        return jsonify({"error": "First and last name are required."}), 400
    if len(pwd) < 8:
        return jsonify({"error": "Password must be at least 8 characters."}), 400
    if pwd != pwd2:
        return jsonify({"error": "Passwords do not match."}), 400

    pwd_hash = generate_password_hash(pwd)
    with get_db() as conn:
        conn.execute(
            "UPDATE users SET first_name=?, last_name=?, password_hash=?, email_verified=1 WHERE email=?",
            (first, last, pwd_hash, email),
        )
        row = conn.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()

    session.pop("pending_email", None)
    session.permanent = True
    session["user_id"] = row["id"]
    return jsonify({"ok": True, "user": current_user()})


@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    """Existing-user login: email + password (after OTP verification)."""
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    pwd = body.get("password") or ""
    if email != session.get("pending_email"):
        return jsonify({"error": "Please verify your email with an OTP first."}), 400

    with get_db() as conn:
        row = conn.execute(
            "SELECT id, password_hash FROM users WHERE email=?", (email,)
        ).fetchone()
    if not row or not row["password_hash"] or not check_password_hash(row["password_hash"], pwd):
        return jsonify({"error": "Incorrect password."}), 401

    session.pop("pending_email", None)
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
    return jsonify({"plan": plan, "name": name})


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


# Initialise DB at import time so gunicorn/wsgi workers also create tables.
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
