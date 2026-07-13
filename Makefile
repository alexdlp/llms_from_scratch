### ------------------------------
### PHONY declarations
### ------------------------------
.PHONY: \
  run_mlflow stop_mlflow status_mlflow \


### ------------------------------
### MLflow UI management
### ------------------------------
run_mlflow:
	./tools/run_mlflow.sh

stop_mlflow:
	tmux kill-session -t mlflow_ui || echo "No mlflow_ui session found"

status_mlflow:
	tmux has-session -t mlflow_ui && echo "MLflow UI is running." || echo "MLflow UI \
	is not running."

### ------------------------------
### Make all sh files in tools executable
### ------------------------------
make-scripts-executable:
	chmod +x tools/*.sh

kill-port:
	@if [ -z "$(filter-out $@,$(MAKECMDGOALS))" ]; then \
		echo "Usage: make kill-port <port>"; \
	else \
		PORT=$(filter-out $@,$(MAKECMDGOALS)); \
		echo "Killing processes on port $$PORT..."; \
		sudo lsof -ti :$$PORT | xargs sudo kill -9 2>/dev/null || echo "No process found on port $$PORT"; \
	fi

%:
	@:

