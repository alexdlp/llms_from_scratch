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
  echo "Warning: .env not found. Environment variables must already be defined."
fi

# --- Conda environment path ---
CONDA_ENV_PATH="$PROJECT_DIR/.venv"

# --- Backend store URI (folder artifacts) ---
BACKEND_URI="file:${PROJECT_DIR}/artifacts"

# --- Check if required variables exist ---
if [ -z "$MLFLOW_HOST" ] || [ -z "$MLFLOW_PORT" ] || [ -z "$MLFLOW_TRACKING_URI" ]; then
  echo "Error: MLFLOW_HOST, MLFLOW_PORT, or MLFLOW_TRACKING_URI not set in environment."
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
  tmux new-session -d -s "$SESSION_NAME" "bash -lc '
    cd \"$PROJECT_DIR\" &&
    source \"\$(conda info --base)/etc/profile.d/conda.sh\" &&
    conda activate \"$CONDA_ENV_PATH\" &&
    mlflow ui \
      --backend-store-uri \"$BACKEND_URI\" \
      --host \"$MLFLOW_HOST\" \
      --port \"$MLFLOW_PORT\"
  '"
  echo "Access MLflow at: $MLFLOW_TRACKING_URI"
  echo "Attach to session: tmux attach -t $SESSION_NAME"
fi