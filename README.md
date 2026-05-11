# BizNova

AI-powered business setup helper for South African entrepreneurs.
Static HTML frontend + Python (Flask) backend that proxies to the OpenAI API.

## Setup

1. Install Python 3.10+
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Configure your API key:
   ```powershell
   copy .env.example .env
   notepad .env   # paste your OPENAI_API_KEY
   ```
   Get a key at https://platform.openai.com/api-keys

## Run

```powershell
python app.py
```

Then open http://localhost:5001

## Endpoints

| Method | Path                  | Purpose                                   |
| ------ | --------------------- | ----------------------------------------- |
| GET    | `/`                   | Serves `biznova_app.html`                 |
| GET    | `/api/health`         | Health + key configuration check          |
| POST   | `/api/chat`           | Nova chat. Body: `{mode, messages[]}`     |
| POST   | `/api/plan`           | Generate a business plan from form fields |
| POST   | `/api/plan/download`  | Returns a Word-compatible `.doc` file     |

## Notes

- Your API key stays on the server — it is **never** sent to the browser.
- `.env` is gitignored. Never put real secrets in `.env.example`.
