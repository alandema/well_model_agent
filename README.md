# Well Model Agent

Well Model Agent is a containerized FastAPI and Streamlit application that uses a LangGraph agent to answer questions about offshore well production and run the Fast Offshore Well Model (FOWM).

The agent uses the free `poolside/laguna-s-2.1:free` model through OpenRouter.

## Current architecture

- **Backend**: FastAPI application served by Uvicorn.
- **Agent**: LangGraph workflow containing a model node and a tool node. The workflow can call the FOWM simulation tool and loops back to the model after tool execution.
- **Model provider**: `langchain-openrouter` with the model and system prompt configured in `backend/agent/prompts/config.json`.
- **FOWM tool**: Integrates the six-state ODE model with SciPy's `odeint`, calculates pressures, and writes simulation results to a CSV file.
- **Conversation persistence**: LangGraph uses a SQLite checkpointer at `backend/.checkpoints/checkpoints.sqlite`, with `thread_id` used to continue a conversation.
- **Frontend**: Streamlit chat UI with sidebar controls for integration time and number of time points.
- **Deployment**: Docker Compose runs the backend and frontend services together.

## Prerequisites

- Docker Desktop with Docker Compose, or Python 3.11+ for local development.
- An OpenRouter API key.

## Configuration

Create `backend/.env` from the checked-in example:

```text
OPENROUTER_API_KEY=your_openrouter_api_key

# Optional LangSmith tracing
LANGSMITH_TRACING_V2=false
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=well-model-agent
```

Do not commit `.env` files or API keys. The repository ignores environment files.

The Streamlit container uses `frontend/.env` to locate the backend:

```text
API_BASE_URL=http://backend:8000
```

For a frontend running directly on the host, use `API_BASE_URL=http://localhost:8000` instead.

The model provider, model ID, temperature, reasoning effort, and system prompt are configured in `backend/agent/prompts/config.json`.

## Run with Docker Compose

From the repository root:

```bash
docker-compose up --build
```

To run in the background:

```bash
docker-compose up --build -d
```

The services are available at:

- **Streamlit frontend**: <http://localhost:8501>
- **FastAPI backend**: <http://localhost:8000>
- **Swagger API documentation**: <http://localhost:8000/docs>
- **ReDoc API documentation**: <http://localhost:8000/redoc>

The Compose configuration mounts the local `backend` and `frontend` directories into the containers, so generated checkpoints and CSV output remain in the project directory during development.

## Run locally without Docker

### Backend

From the repository root, create or activate a Python environment, install the backend dependencies, and start FastAPI:

```bash
cd backend
pip install -r requirements.txt
uvicorn api.main:app --reload
```

The backend loads `backend/.env` and listens on <http://localhost:8000>.

### Frontend

In a second terminal:

```bash
cd frontend
pip install -r requirements.txt
```

Set `API_BASE_URL=http://localhost:8000` in `frontend/.env`, then start Streamlit:

```bash
streamlit run app/main.py
```

Open <http://localhost:8501> in a browser.

## API

### `GET /`

Returns a basic application greeting:

```json
{"message": "Hello World"}
```

### `GET /health`

Returns the backend health status:

```json
{"status": "ok"}
```

### `POST /chat`

Runs the agent and returns the completed response. `integration_time` and `time_points` are required because they are passed to the FOWM tool through the LangGraph runtime configuration.

Request:

```json
{
  "message": "Simulate the well with the default model parameters.",
  "thread_id": null,
  "integration_time": 100000,
  "time_points": 1001
}
```

Response:

```json
{
  "response": "...",
  "thread_id": "generated-or-existing-thread-id",
  "interrupt": null
}
```

Use the returned `thread_id` in subsequent requests to preserve the LangGraph conversation state. The `interrupt` field is populated if a tool execution pauses for human approval.

Example using PowerShell:

```powershell
$body = @{
  message = "Simulate the well with the default model parameters."
  integration_time = 100000
  time_points = 1001
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri http://localhost:8000/chat `
  -ContentType "application/json" -Body $body
```

### `POST /chat/stream`

Streams generated text as `text/plain`. After the response text, the endpoint emits a metadata line beginning with `__META__:` containing the `thread_id` and, when applicable, an interrupt payload.

## FOWM simulation

The `fowm_model` tool accepts the following model inputs:

- Initial state `x0` (optional; a default six-state initial condition is used when omitted).
- Gas injection rate.
- Choke opening percentage.
- Separator pressure.
- Reservoir pressure.
- FOWM parameters `mlstill`, `Cg`, `Cout`, `Veb`, `E`, `Kw`, `Ka`, and `Kr`.

The integration controls are supplied by the API request:

- `integration_time` must be greater than zero.
- `time_points` must be at least `2`.

Each successful simulation writes a CSV file containing time, six state variables, and the calculated `Ppdg`, `Ptt`, `Prt`, and `Prb` pressures to `backend/.outputs/`.

## Project structure

```text
well_model_agent/
├── docker-compose.yml
├── README.md
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── .env.example
│   ├── agent/
│   │   ├── graph.py                 # LangGraph workflow and SQLite checkpointer
│   │   ├── state.py                 # Agent message state
│   │   ├── prompts/config.json      # OpenRouter model configuration
│   │   ├── services/
│   │   │   ├── config.py
│   │   │   ├── llm_model_factory.py
│   │   │   └── model_runner.py      # SciPy ODE integration helper
│   │   └── tools/fowm.py             # FOWM equations and LangChain tool
│   └── api/main.py                   # FastAPI routes
├── frontend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/main.py                   # Streamlit chat application
└── fowm_code.py                     # Original standalone/reference FOWM script
```

Generated runtime data is stored in ignored directories:

- `backend/.checkpoints/` — SQLite conversation checkpoints.
- `backend/.outputs/` — FOWM CSV results.

## Stop the application

```bash
docker-compose down
```

To also remove Compose-managed volumes:

```bash
docker-compose down -v
```