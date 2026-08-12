#!/usr/bin/env python3
"""
Live Progress Monitor & Terminal Dashboard for EBV Knowledge System Queue.
Parses Pueue task statuses, active log progress, paper counts, and system metrics.
"""

import os
import sys
import time
import json
import re
import argparse
import subprocess
from pathlib import Path


def get_pueue_status():
    """Fetch structured JSON status from Pueue CLI."""
    pueue_bin = os.getenv("PUEUE_BIN", "/storage/harsha_projects/server_environments/bin/pueue")
    if not os.path.exists(pueue_bin):
        pueue_bin = "pueue"
    try:
        res = subprocess.check_output([pueue_bin, "status", "--json"], stderr=subprocess.DEVNULL)
        return json.loads(res.decode("utf-8"))
    except Exception as e:
        return {"error": str(e), "tasks": {}}


def parse_task_log(task_id, pueue_bin):
    """Parse log for a running task to extract step, query, and paper progress."""
    res = ""
    log_file = Path.home() / ".local/share/pueue/task_logs" / f"{task_id}.log"
    if log_file.exists():
        try:
            res = log_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            pass

    if not res:
        try:
            res = subprocess.check_output(
                [pueue_bin, "log", str(task_id)], stderr=subprocess.DEVNULL
            ).decode("utf-8", errors="ignore")
        except Exception:
            return {"step": "Starting", "processed": 0, "total": 0, "current_file": "N/A", "pct": 0.0}



    # Extract Scraper total found
    total_found = 0
    m_found = re.findall(r"Scraper found (\d+) articles", res)
    if m_found:
        total_found = int(m_found[-1])

    # Extract XML files parsed
    xml_files = re.findall(r"Parsing JATS XML file: .*/xml/(\w+\.xml)", res)
    pdf_files = re.findall(r"Parsing PDF file: .*/pdf/(\w+\.pdf)", res)
    processed_count = len(xml_files) + len(pdf_files)
    last_file = xml_files[-1] if xml_files else (pdf_files[-1] if pdf_files else "N/A")

    # Detect Step
    step = "PubMed Scraping"
    if "Step 4: Materialize to Neo4j" in res:
        step = "Neo4j Materialization"
    elif "Step 3: Index Chunks to LanceDB" in res:
        step = "LanceDB Vector Indexing"
    elif "Step 2: Parse and Map Documents" in res or xml_files or pdf_files:
        step = "NER & Entity Mapping"
    elif "Step 1: Scraper initialization" in res:
        step = "PubMed Scraping"

    pct = (processed_count / total_found * 100.0) if total_found > 0 else 0.0
    return {
        "step": step,
        "processed": processed_count,
        "total": total_found,
        "current_file": last_file,
        "pct": min(pct, 100.0),
    }



def draw_progress_bar(pct, width=25):
    """Generate ASCII progress bar."""
    filled = int(width * pct / 100.0)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct:5.1f}%"


def render_dashboard(watch=False):
    """Render terminal status dashboard."""
    pueue_bin = os.getenv("PUEUE_BIN", "/storage/harsha_projects/server_environments/bin/pueue")
    if not os.path.exists(pueue_bin):
        pueue_bin = "pueue"

    while True:
        data = get_pueue_status()
        tasks = data.get("tasks", {})

        running_tasks = []
        queued_tasks = []
        finished_tasks = []

        for tid, t in sorted(tasks.items(), key=lambda x: int(x[0])):
            status_obj = t.get("status", {})
            cmd = t.get("command", "")
            m_q = re.search(r'--query\s+[\"\']?(.*?)[\"\']?\s+--', cmd + " --")
            query = m_q.group(1).strip('"\'') if m_q else "General Ingestion"

            if "Running" in status_obj:
                running_tasks.append((tid, query, t))
            elif "Queued" in status_obj:
                queued_tasks.append((tid, query, t))
            elif "Done" in status_obj:
                finished_tasks.append((tid, query, t))

        if watch:
            os.system("clear" if os.name == "posix" and os.getenv("TERM") else "cls" if os.name == "nt" else "")

        print("==================================================================================")
        print("                 🔬 EBV KNOWLEDGE SYSTEM - QUEUE PROGRESS MONITOR                 ")
        print("==================================================================================")
        print(f" Time: {time.strftime('%Y-%m-%d %H:%M:%S CEST')} | Host: rinamochana")
        print(f" Active Workers: {len(running_tasks)} | Queued Tasks: {len(queued_tasks)} | Total History: {len(tasks)}")
        print("----------------------------------------------------------------------------------")

        print("\n⚡ ACTIVE WORKER SLOTS & FILE PROCESSING PROGRESS:")
        if not running_tasks:
            print("  (No active worker tasks currently running)")
        else:
            print(f" {'ID':<6} | {'Query Search Topic':<30} | {'Step / Status':<22} | {'File Progress':<18}")
            print("-" * 84)
            for tid, query, _ in running_tasks[:20]:
                log_info = parse_task_log(tid, pueue_bin)
                q_short = query[:28] + ".." if len(query) > 30 else query
                if log_info['total'] > 0:
                    prog_str = f"{log_info['processed']}/{log_info['total']} ({log_info['pct']:.0f}%)"
                elif log_info['processed'] > 0:
                    prog_str = f"{log_info['processed']} papers"
                else:
                    prog_str = "Scraping..."

                print(f" #{tid:<5} | {q_short:<30} | {log_info['step']:<22} | {prog_str:<18}")
                if log_info["current_file"] != "N/A":
                    print(f"        └─ Current Paper: {log_info['current_file']} {draw_progress_bar(log_info['pct'], width=15)}")

        print("\n📋 RECENTLY QUEUED TOPICS (UPCOMING):")
        if not queued_tasks:
            print("  (Queue empty)")
        else:
            for tid, query, _ in queued_tasks[:5]:
                print(f"  • Task #{tid}: \"{query}\"")
            if len(queued_tasks) > 5:
                print(f"  ... and {len(queued_tasks) - 5} more queued tasks.")

        print("==================================================================================")
        if not watch:
            break
        print(" Press Ctrl+C to exit monitoring mode. Refreshing in 3 seconds...")
        time.sleep(3)


def main():
    parser = argparse.ArgumentParser(description="EBV KG Ingestion Queue Monitor")
    parser.add_argument("--watch", "-w", action="store_true", help="Continuously poll and update dashboard")
    args = parser.parse_args()
    render_dashboard(watch=args.watch)


if __name__ == "__main__":
    main()
