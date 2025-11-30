# Stateful Router

## Project Overview

The Stateful Router is a hierarchical neuro-symbolic architecture designed to create verifiable chains of thought in AI reasoning. This implementation separates strategic planning from tactical execution, maintains comprehensive audit trails, and provides confidence tracking throughout the reasoning process.

## Architecture Components

1. **Strategic Planner** - High-level problem decomposition and strategy selection
2. **Tactical Router** - Step-by-step execution of strategic plans
3. **Expert Toolkit** - Deterministic and probabilistic tools for specific operations
4. **Verification Module** - Continuous consistency checking and confidence tracking
5. **Hierarchical Memory** - Multi-level logging system for complete auditability

## Directory Structure

```
stateful_router_v1/
│
├── stateful_router/              # The main Python package
│   ├── __init__.py               # Makes the folder a package
│   ├── config/                   # Configuration (prompts, settings)
│   │   ├── __init__.py
│   │   ├── prompts.py
│   │   └── settings.py
│   ├── core/                     # Core logic (Orchestrator, Planners)
│   │   ├── __init__.py
│   │   ├── memory.py
│   │   ├── orchestrator.py
│   │   ├── strategic_planner.py
│   │   ├── tactical_router.py
│   │   └── verification.py
│   └── tools/                    # Expert tools (add, parse, etc.)
│       ├── __init__.py
│       ├── base.py
│       ├── deterministic.py
│       ├── probabilistic.py
│       └── registry.py
│
├── .env                          # local file for API keys
├── .gitignore
├── README.md                     # This file
├── requirements.txt
├── run_test.py                   # Quick Start script
├── setup.py                      # Makes the project installable
└── venv/                         # virtual environment

```

## Installation

1.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
2.  **Install all dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Install the project in "editable" mode:**
    (This is what allows `from stateful_router...` to work)
    ```bash
    pip install -e .
    ```

## Configuration (Required)

The system will not work without API keys.

1.  Create a file named `.env` in the root of the project (`stateful_router_v1/`).
2.  Copy and paste the following, adding your secret API keys:

    ```ini
    # --- API Keys (REQUIRED) ---
    # Get from: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
    OPENAI_API_KEY="sk-..."
    
    # Get from: [https://console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys)
    ANTHROPIC_API_KEY="sk-..."
    ```
The system (`config/settings.py`) will automatically load these keys.

## Quick Start

Your repository includes a `run_test.py` script. After you have installed the dependencies (Step 4) and configured your `.env` file (Step 5), you can run the system.

```bash
source venv/bin/activate

python run_test.py
```

## License

MIT License
