# Running MediBot

Three terminals, or run them one at a time.

## 1. Ingest documents (once)

```powershell
python backend/scripts/ingest.py --recreate
```

## 2. Backend

```powershell
python run_api.py
```
-> http://localhost:8000/docs

## 3. Frontend

```powershell
cd frontend
npm install      # first time only
npm run dev
```
-> http://localhost:3000

## Order matters

Embedded Qdrant allows ONE process to hold the storage folder. Stop the
backend before re-running ingest.

The frontend needs the backend running, or login will report it can't reach
the API.

## Demo logins

Click any row on the login screen, or type them:

| Username | Password | Role |
|---|---|---|
| dr.mehta | doctor123 | doctor |
| nurse.priya | nurse123 | nurse |
| billing.ravi | billing123 | billing_executive |
| tech.anand | tech123 | technician |
| admin.sys | admin123 | admin |
