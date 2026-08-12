#!/usr/bin/env bash
# EBV Knowledge System Queue & File Ingestion Progress Monitor
# Usage: ./script.sh [--watch]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if running remotely on rinamochana or locally
if [ -d "/storage/harsha_projects/ebv_KG" ]; then
    /home/harsha/ebv_KG_venv/bin/python "${SCRIPT_DIR}/scripts/monitor_progress.py" "$@"
else
    # Execute on rinamochana via SSH
    ssh -t rinamochana "/home/harsha/ebv_KG_venv/bin/python /storage/harsha_projects/ebv_KG/scripts/monitor_progress.py $@"
fi
