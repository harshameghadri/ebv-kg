# Agent Handover Document

> **Instructions for Agents:** 
> Update this document at the end of your session if you are handing off a task to another model or suspending work. Ensure the next agent can immediately resume where you left off.

## 1. Current State
- Project configuration initialized: `pyproject.toml` created, virtual environment set up, and packages installed via `uv sync`.
- Core architecture spec gaps filled in `ebv-rag-engineering-spec.md` (completed `T701` and `T702` for Storage, Scaling, Security, Monitoring, and Risk).
- Table of Contents in the specification updated to remove raw draft warning markers.
- Git repository updated and changes committed.
- **8-Task Parallelism Active**: Spawned 8 concurrent background subagents to work on core modules.

## 2. Active Tasks (In Progress)
- **T101**: Implement PMC XML parser (in `app/ingestion/`).
  - *Status*: Subagent `48f44395-a3be-495a-b965-4cacb16d0a81` (Role: `Ingestion Engineer`).
- **T102**: Implement PDF extractor (in `app/ingestion/`).
  - *Status*: Subagent `c1abb4cb-0435-43df-87d6-73672a07908b` (Role: `PDF Extraction Engineer`).
- **T103**: Implement PubMed API scraper (in `app/ingestion/`).
  - *Status*: Subagent `b8a3f8d6-2a40-4464-b49e-6817d7a3decd` (Role: `Data Fetching Engineer`).
- **T104**: Implement GEO/SRA metadata crawler (in `app/ingestion/`).
  - *Status*: Subagent `357902df-4567-43ee-b5c0-ef128ce7c369` (Role: `GEO Crawler Engineer`).
- **T201**: Implement SciSpacy & Bern2 API NER wrapper (in `app/processing/`).
  - *Status*: Subagent `e022028f-98df-435a-808e-c02fa5a475c1` (Role: `NER NLP Engineer`).
- **T202**: Implement local dictionary-based synonym resolver (in `app/processing/`).
  - *Status*: Subagent `2687837a-2b65-4dd4-b08c-8f7f2503017f` (Role: `Bioinformatics Ontology Engineer`).
- **T204**: Create PostgreSQL database schema (in `app/database/`).
  - *Status*: Subagent `e2d90e41-2ecf-4531-9042-cb0855782fcb` (Role: `Database Architect`).
- **T301**: Implement Neo4j Graph DB client wrapper (in `app/materialization/`).
  - *Status*: Subagent `1ccc34ae-c82a-415c-bd4e-9b5d47a36443` (Role: `Graph Database Engineer`).

## 3. Next Steps (Immediate)
- Wait for subagents to execute their tasks. To conserve tokens and follow user instructions, we check in on running tasks only every 3-5 minutes, rather than spamming checkin requests.
- Verify subagent outputs and run test suites when notifications arrive.

## 4. Pending Blockers or Open Questions
- None.

## 5. Important Context / Gotchas
- The local Python environment is set up under `.venv/` (Python 3.12.10) with `uv` package manager.
- All subagents have been instructed to ensure clean interfaces and keep final system linking in mind when structuring their schemas and function outputs.
- Always conform to `AgentBehavior.md` and check `Gemini.md` directives.
