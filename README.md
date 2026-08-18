# Well Model Agent

Well Model Agent is a containerized FastAPI and Streamlit application that uses a LangGraph agent to answer questions about offshore well production and run the Fast Offshore Well Model (FOWM).

The agent uses the `poolside/laguna-s-2.1:free` model (because it is free - Using a better model will definitely make it better.) through OpenRouter. The configured model uses temperature `0` and high reasoning effort; see `backend/agent/prompts/config.json`.

## Current architecture

- **Backend**: FastAPI application served by Uvicorn.
- **Agent**: LangGraph workflow containing model, tool, and stop nodes. It can call the FOWM simulation, web search, CSV analysis, CSV reading, and terminal tools, then loops back to the model after tool execution. The workflow stops after more than three tool rounds for a single user request.
- **Model provider**: `langchain-openrouter` with the model and system prompt configured in `backend/agent/prompts/config.json`.
- **FOWM tool**: Integrates the six-state ODE model with SciPy's `odeint`, converts physical inputs with Pint, calculates pressures, and writes simulation results to a CSV file.
- **Conversation persistence**: LangGraph uses a SQLite checkpointer at `backend/agent/.checkpoints/checkpoints.sqlite`, with `thread_id` used to continue a conversation.
- **Frontend**: Streamlit chat UI that calls `POST /chat`. It keeps the current `thread_id` in Streamlit session state.
- **Deployment**: Docker Compose runs the backend and frontend services together.

## Prerequisites

- Docker Desktop with Docker Compose, or Python 3.11+ for local frontend development. The backend image currently uses Python 3.14; use a compatible Python version when running the backend outside Docker.
- An OpenRouter API key.
- A Tavily API key is optional and enables the web-search tool.

## Configuration

Create `backend/.env` from `backend/.env.example`:

```text
OPENROUTER_API_KEY=your_openrouter_api_key

# Optional web search
TAVILY_API_KEY=your_tavily_api_key

# Optional LangSmith tracing
LANGSMITH_TRACING_V2=
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=
```

Do not commit `.env` files or API keys. The repository ignores environment files.

The Streamlit container uses `frontend/.env` to locate the backend:

```text
API_BASE_URL=http://backend:8000
```

For a frontend running directly on the host, use `API_BASE_URL=http://localhost:8000` instead.

The model provider, model ID, temperature, reasoning effort, and system prompt are configured in `backend/agent/prompts/config.json`. `LANGSMITH_TRACING_V2`, `LANGSMITH_API_KEY`, and `LANGSMITH_PROJECT` are documented optional settings, but this application does not explicitly initialize LangSmith; configure tracing according to the LangChain/LangSmith environment-variable behavior if you use it.

## Run with Docker Compose

From the repository root:

```bash
docker-compose up --build
```

To run in the background:

```bash
docker-compose up --build -d
```

Telemetry is stored in the host directory `./data/telemetry.db`. Compose bind
mounts that directory at `/data` in `control_room` and `backend`,
so the control-room producer, Streamlit reader, and future container readers
use the same SQLite file. The database can also be queried from the host:

```powershell
sqlite3 .\data\telemetry.db "SELECT well_id, COUNT(*) FROM sensor_readings GROUP BY well_id;"
```

The database uses SQLite WAL mode for short concurrent reads and writes. Keep
the database on the same local host filesystem; do not place this SQLite file
on a network share.

The services are available at:

- **Streamlit frontend**: <http://localhost:8501>
- **FastAPI backend**: <http://localhost:8000>
- **Swagger API documentation**: <http://localhost:8000/docs>
- **ReDoc API documentation**: <http://localhost:8000/redoc>

The Compose configuration mounts the local `backend` and `frontend` directories into the containers. The backend writes checkpoints to `/app/agent/.checkpoints` and FOWM CSV files to `/app/agent/.outputs`, which correspond to `backend/agent/.checkpoints` and `backend/agent/.outputs` on the host. These directories are ignored by Git.

## Run locally without Docker

### Backend

From the repository root, create or activate a Python environment, install the backend dependencies, and start FastAPI:

```bash
cd backend
pip install -r requirements.txt
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The backend image and local process both expect imports to resolve from `backend`; run the command from that directory. The application reads environment variables from the process environment; the repository does not currently load `backend/.env` itself, so export the variables or use an environment loader before starting it locally. It listens on <http://localhost:8000>.

### Frontend

In a second terminal:

```bash
cd frontend
pip install -r requirements.txt
```

Create `frontend/.env` from `frontend/.env.example`. Set `API_BASE_URL=http://localhost:8000` for a host-run frontend (the example currently contains `localhost:8000` without the URL scheme), then start Streamlit:

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

Runs the agent and returns the completed response. The agent selects tool parameters, including FOWM simulation settings, from the conversation. The request body contains only the user message and optional conversation thread ID.

Request:

```json
{
  "message": "Simulate the well with the default model parameters.",
  "thread_id": null
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

Use the returned `thread_id` in subsequent requests to preserve the LangGraph conversation state. The response model includes an `interrupt` field, but the current `/chat` implementation does not populate it from graph state. Human-in-the-loop resume support is defined internally but is not exposed as a route, so approval is not currently available through the public API.

Example using PowerShell:

```powershell
$body = @{
  message = "Simulate the well with the default model parameters."
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri http://localhost:8000/chat `
  -ContentType "application/json" -Body $body
```

### `POST /chat/stream`

Runs the same graph as `/chat` and returns `text/plain`. The endpoint currently yields the completed response as one chunk rather than token-by-token. After the response, it emits a metadata line beginning with `__META__:` containing JSON with the `thread_id`. The current implementation does not include an interrupt payload in this metadata.

## FOWM simulation

The `fowm_model` tool accepts the following inputs. Physical values are objects with a numeric `value` and a Pint-compatible `unit`:

- Initial state `x0` (optional; a default six-state initial condition is used when omitted).
- `gas_injection_rate` (default `165000 m^3/day`).
- `choke_opening` (default `16 percent`).
- `separator_pressure` (default `101325 Pa`).
- `reservoir_pressure` (default `2.25e7 Pa`).
- FOWM parameters `mlstill`, `Cg`, `Cout`, `Veb`, `E`, `Kw`, `Ka`, and `Kr`, each with a default defined in `backend/agent/tools/fowm_model.py`.

Simulation controls are also parameters of the `fowm_model` tool:

- `integration_time` must be greater than zero.
- `time_points` must be at least `2`.

Both controls default to `100000` seconds and `1001` time points when the agent does not specify them. They are not direct frontend or API request fields; they are tool arguments selected by the agent.

Each successful simulation writes a timestamped CSV file containing `t`, six state variables (`x1` through `x6`), and the calculated `Ppdg`, `Ptt`, `Prt`, and `Prb` pressures to `backend/agent/.outputs/`. The tool returns the absolute path inside the container, for example `/app/agent/.outputs/fowm_YYYY-MM-DD_HH-MM-SS.csv`. The `summarize_csv` and `read_csv` tools accept these container paths.

The other registered tools are:

- `summarize_csv` — summarizes numeric CSV columns.
- `read_csv` — reads selected CSV columns and rows as a Markdown table.
- `web_search` — searches Tavily when `TAVILY_API_KEY` is configured.
- `terminal` — executes shell commands from the backend process with a 30-second timeout; use this capability cautiously.

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
│   │   └── tools/
│   │       ├── fowm_model.py         # FOWM equations and LangChain tool
│   │       ├── read_csv.py
│   │       ├── summarize_csv.py
│   │       ├── terminal.py
│   │       ├── unit_conversion.py
│   │       └── web_search.py
│   └── api/main.py                   # FastAPI routes
├── frontend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/main.py                   # Streamlit chat application
└── fowm_code.py                     # Original standalone/reference FOWM script
```

Generated runtime data is stored in ignored directories:

- `backend/agent/.checkpoints/` — SQLite conversation checkpoints.
- `backend/agent/.outputs/` — FOWM CSV results.

## Stop the application

```bash
docker-compose down
```

To also remove Compose-managed volumes:

```bash
docker-compose down -v
```