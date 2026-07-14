#!/bin/bash
set -e

SESSION_NAME="mlflow_ui"

# --- Resolve project directory (parent of tools/) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$PROJECT_DIR/.env"

# --- Load .env directly into environment ---
if [ -f "$ENV_FILE" ]; then
  set -a
  source "$ENV_FILE"
  set +a
  echo "Loaded environment variables from $ENV_FILE"
else
  echo "Warning: .env not found. Using local MLflow defaults."
fi

# --- Local defaults ---
MLFLOW_HOST="${MLFLOW_HOST:-127.0.0.1}"
MLFLOW_PORT="${MLFLOW_PORT:-5001}"
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://${MLFLOW_HOST}:${MLFLOW_PORT}}"
MLFLOW_BIN="$PROJECT_DIR/.venv/bin/mlflow"

# --- Backend store URI (folder artifacts) ---
BACKEND_URI="file:${PROJECT_DIR}/artifacts"

# --- Check local MLflow installation ---
if [ ! -x "$MLFLOW_BIN" ]; then
  echo "Error: MLflow executable not found at $MLFLOW_BIN. Run 'uv sync' first."
  exit 1
fi

# --- Check if port is free ---
if lsof -i :"$MLFLOW_PORT" &>/dev/null; then
  echo "Error: port $MLFLOW_PORT is already in use."
  exit 1
fi

# --- Launch MLflow in tmux ---
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "MLflow UI is already running in tmux session '$SESSION_NAME'."
else
  echo "Starting MLflow UI on $MLFLOW_TRACKING_URI"
  tmux new-session -d -s "$SESSION_NAME" "
    cd \"$PROJECT_DIR\" &&
    exec \"$MLFLOW_BIN\" ui \
      --backend-store-uri \"$BACKEND_URI\" \
      --host \"$MLFLOW_HOST\" \
      --port \"$MLFLOW_PORT\"
  "
  echo "Access MLflow at: $MLFLOW_TRACKING_URI"
  echo "Attach to session: tmux attach -t $SESSION_NAME"
fi
