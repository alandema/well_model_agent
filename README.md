# well_model_agent

A minimal LangGraph + FastAPI + Streamlit application that connects to Google's Gemini model.

## Architecture

- **Backend**: FastAPI with a single-node LangGraph that calls Gemini via `langchain-google-genai`
- **Frontend**: Streamlit chat interface that communicates with the backend
- **Docker Compose**: Orchestrates both services

## Prerequisites

- Docker and Docker Compose installed
- A Google Gemini API key (get one at [Google AI Studio](https://aistudio.google.com/app/apikey))

## Quick Start

### 1. Set up environment variables

Create a `.env` file in the `backend` directory:

```bash
cp backend/.env.example backend/.env  # if example exists, or create manually
```

Add your Google API key to `backend/.env`:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 2. Build and run with Docker Compose

```bash
# Build and start both services
docker-compose up --build

# Or run in background
docker-compose up --build -d
```

### 3. Access the application

- **Frontend (Streamlit)**: http://localhost:8501
- **Backend API (FastAPI)**: http://localhost:8000
- **API Docs (Swagger)**: http://localhost:8000/docs

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root endpoint |
| GET | `/health` | Health check |
| POST | `/chat` | Chat with Gemini |

### Chat Endpoint Example

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, how are you?"}'
```

Response:
```json
{"response": "Hello! I'm doing well, thank you for asking..."}
```

## Development (without Docker)

### Backend

```bash
cd backend
pip install -r requirements.txt
# Set GOOGLE_API_KEY in your environment
uvicorn api.main:app --reload
```

### Frontend

```bash
cd frontend
pip install -r requirements.txt
streamlit run app/main.py
```

## Project Structure

```
well_model_agent/
├── docker-compose.yml
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── .env              # Add your GOOGLE_API_KEY here
│   ├── agent/
│   │   └── graph.py      # LangGraph single-node graph
│   └── api/
│       └── main.py       # FastAPI app with /chat endpoint
└── frontend/
    ├── Dockerfile
    ├── requirements.txt
    ├── .env              # Optional: API_BASE_URL
    └── app/
        └── main.py       # Streamlit chat interface
```

## Stopping the Application

```bash
# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```