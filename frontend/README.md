# BTCUSDT Prediction Frontend

React + TypeScript frontend for the BTCUSDT 5-minute short-term prediction dashboard.

## Commands

```bash
npm install
npm run dev
npm run build
```

The dev server defaults to:

```text
http://127.0.0.1:5173/
```

## Backend API

All API calls are centralized in:

```text
src/services/api.ts
```

The frontend only requests project-owned endpoints under `/api/...`.
It does not request or scrape AICoin chart pages and it does not implement order execution.

Set a backend base URL with:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

When backend endpoints are unavailable, the API layer returns mock data from:

```text
src/mocks/marketMock.ts
```

Run the local Python API from the project root with:

```bash
uvicorn api.app:app --app-dir src --host 127.0.0.1 --port 8000
```
