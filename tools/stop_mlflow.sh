#!/bin/bash
set -e

SESSION_NAME="mlflow_ui"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$PROJECT_DIR/.env"

if [ -f "$ENV_FILE" ]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

MLFLOW_HOST="${MLFLOW_HOST:-127.0.0.1}"
MLFLOW_PORT="${MLFLOW_PORT:-5001}"

MLFLOW_PIDS="$(lsof -tiTCP:"$MLFLOW_PORT" -sTCP:LISTEN 2>/dev/null || true)"
VALID_PIDS=""

for PID in $MLFLOW_PIDS; do
  PROCESS_COMMAND="$(ps -p "$PID" -o command= 2>/dev/null || true)"

  if [[ "$PROCESS_COMMAND" == *"$PROJECT_DIR/.venv/bin/python"*gunicorn*"$MLFLOW_HOST:$MLFLOW_PORT"* ]]; then
    VALID_PIDS="$VALID_PIDS $PID"
  elif [ -n "$PROCESS_COMMAND" ]; then
    echo "Warning: PID $PID does not belong to this MLflow server; leaving it untouched."
  fi
done

if [ -n "$VALID_PIDS" ]; then
  echo "Stopping MLflow processes:$VALID_PIDS"
  kill -TERM $VALID_PIDS
fi

tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

for _ in {1..50}; do
  if ! lsof -nP -iTCP:"$MLFLOW_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "MLflow UI stopped. Port $MLFLOW_PORT is free."
    exit 0
  fi
  sleep 0.1
done

echo "Error: MLflow did not release port $MLFLOW_PORT."
exit 1
