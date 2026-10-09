#!/bin/bash
# One-time setup before the first `docker compose up`.
set -e

# Folders Airflow mounts into the containers
mkdir -p ./logs ./plugins ./config ./dags/model ./dags/reports

# .env is read by docker compose: run containers as your user and install ML packages
cat > .env << EOF
AIRFLOW_UID=$(id -u)
_PIP_ADDITIONAL_REQUIREMENTS=scikit-learn kneed
EOF

echo "Setup complete. Contents of .env:"
cat .env
