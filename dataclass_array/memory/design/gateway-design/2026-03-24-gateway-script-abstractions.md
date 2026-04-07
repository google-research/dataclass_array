# Design: Unifying Orchestration & Build Abstractions

## Goal
Clean up and refactor the existing script orchestration (`run_agent_manager.py` and `rebuild.sh`) into a unified, robust Python-based framework suited for multi-workspace Gateway lifecycle triggers.

---

## 1. Dimensional Weaknesses in Current setup

*   **Single-Tenant Hardcoding**: `rebuild.sh` hardcodes absolute paths like `GOB="/path/to/agents"` which breaks parallel environment execution pointers.
*   **Language Splitting**: Lifecycle logic is bifurcated between Python (orchestration) and Bash (building/patching). This fragments error propagation triggers.
*   **Hardcoded Patch Matrix**: `rebuild.sh` maintains a rigid line-by-line `git apply` checklist. Adding a plugin requires manually editing the bash script.

---

## 2. Refactored Layout: Pure Python Orchestrator

Instead of standalone scripts, we consolidate logic into a structured Python execution package:

```text
v1/scripts/
├── agent_manager.py               # Unified CLI (replaces run_agent_manager.py)
└── backend/
    ├── __init__.py
    ├── builder.py                # **Absorbs rebuild.sh** (Worktrees & Compile)
    ├── patches.py                # Data-driven Patch Registry
    ├── process.py                # TMUX & Backend lifecycle monitor
    └── config.py                 # Path resolution & environment context
```

---

## 3. Key Technical Updates

### A. Absorbing `rebuild.sh` Into Python (`builder.py`)
Running `git` and `npm` triggers in Python via `subprocess` allows for robust try/except crash handling. This provides safe cleanup of `git worktree` volumes if build phases fail mid-flight.

### B. Data-Driven Patch Registry (`patches.py`)
Move away from static checklists:
Create a JSON/Python struct manifest indexing how patches connect:
```python
PATCH_MANIFEST = [
    {
        "name": "main_frame",
        "source": "web/main.tsx.patch",
        "target": "exa/agent_ui_toolkit/dev/main.tsx"
    },
    # Plugins can register their own automatically
]
```
*   **Plugin injection**: The builder scans relevant plugin folders, reads local manifests, and appends them dynamically allowing plug-and-play iteration.

### C. Unified CLI Design (`agent_manager.py`)
Implement clean subcommand architectures using `argparse`:

*   `agent_manager.py start [--gateway]`: Coordinates active stack execution.
*   `agent_manager.py build`: Triggers isolated builds manually for diagnostics.
*   `agent_manager.py list`: Reads the state nodes displaying currently deployed triggers.

---

## 4. Step-by-Step Migration Vector

1.  **Context Phase**: Assemble `config.py` loading relative paths making environment pointers agnostic.
2.  **Porting Phase**: Convert sequential patch checkpoint filters from `.sh` routines into `.py` loops.
3.  **Worktree Hook**: Inject trigger isolates allowing continuous execution safely wrapped in try/catch wrappers.
