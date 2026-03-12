# Dependencies & Setup

## Backend (Python)

Install with:
```
pip install -r backend/requirements.txt
```

| Package | Purpose |
|---|---|
| `fastapi` | Web framework / API |
| `uvicorn[standard]` | ASGI server to run FastAPI |
| `python-multipart` | File upload support |
| `pydantic` | Data validation/schemas |
| `opencv-python` | Image/video processing |
| `numpy` | Array operations |
| `easyocr` | OCR for reading plate text |
| `ultralytics` | YOLOv8 plate detection |
| `Pillow` | Image handling |

Run the backend:
```
cd backend
python -m uvicorn main:app --host localhost --port 8000
```

---

## Frontend (Node.js)

Requires: **Node.js** (v18+) and **npm**

Install with:
```
cd website
npm install
```

Key packages (all handled automatically by npm):
- `react` + `react-dom` — UI framework
- `vite` — dev server & bundler
- `tailwindcss` — styling
- `framer-motion` — animations
- `lucide-react` — icons
- `@radix-ui/*` — UI components (shadcn/ui)
- `@tanstack/react-query` — data fetching
- `react-router-dom` — routing

Run the frontend:
```
cd website
npm run dev
```

Opens at **http://localhost:8080**

---

## Quick Start

Terminal 1:
```
cd backend
python -m uvicorn main:app --host localhost --port 8000
```

Terminal 2:
```
cd website
npm run dev
```
