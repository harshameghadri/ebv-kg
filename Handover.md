# Agent Handover Document

> **Instructions for Agents:** 
> Update this document at the end of your session if you are handing off a task to another model or suspending work. Ensure the next agent can immediately resume where you left off.

## 1. Current State
- Project configuration initialized: `pyproject.toml` created, virtual environment set up, and packages installed via `uv sync`.
- Core architecture spec gaps filled in `ebv-rag-engineering-spec.md` (completed `T701` and `T702` for Storage, Scaling, Security, Monitoring, and Risk).
- Table of Contents in the specification updated to remove raw draft warning markers.
- Subagents spawned for parallel coding tasks.

## 2. Active Tasks (In Progress)
- **T101**: Implement PMC XML parser (in `app/ingestion/`).
  - *Status*: Running in background subagent `48f44395-a3be-495a-b965-4cacb16d0a81` (Role: `Ingestion Engineer`, model: `flash`).
- **T204**: Create PostgreSQL database schema defining raw source data, parsed chunks, extracted entities, and curation tables.
  - *Status*: Running in background subagent `e2d90e41-2ecf-4531-9042-cb0855782fcb` (Role: `Database Architect`, model: `flash`).

## 3. Next Steps (Immediate)
- Wait for subagents to report completion or request input.
- Review their generated files:
  - `app/ingestion/pmc_parser.py` and `tests/test_pmc_parser.py` (Ingestion Engineer).
  - `app/database/schema.sql`, `app/database/schema.py`, and `tests/test_schema.py` (Database Architect).
- Once completed, run the test suites to verify.

## 4. Pending Blockers or Open Questions
- None.

## 5. Important Context / Gotchas
- The local Python environment is set up under `.venv/` (Python 3.12.10) with `uv` package manager.
- Always conform to `AgentBehavior.md` and check `Gemini.md` directives.
