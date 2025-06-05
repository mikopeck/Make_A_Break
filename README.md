# Make_A_Break: LLM Jailbreak Testing Workbench

**Make_A_Break** is a Python-based tool designed for researchers and developers to automatically test Large Language Model (LLM) vulnerabilities by attempting various "jailbreak" techniques. It utilizes local Ollama models for both the target LLM and an AI Judge, LangGraph for orchestrating the testing workflow, and Streamlit for an interactive user interface.

**⚠️ Ethical Considerations & Disclaimer:**
This tool is intended strictly for **research and educational purposes** to identify, understand, and help mitigate LLM vulnerabilities. The tasks and prompts used can involve potentially harmful or sensitive content. Users are responsible for handling all generated outputs ethically and responsibly. Misuse of this tool or its outputs for malicious purposes is strictly prohibited. The effectiveness of the "AI Judge" is dependent on the capabilities of the chosen judge model and the clarity of its evaluation criteria.

---
## Features

* **Local LLM Interaction:** Leverages [Ollama](https://ollama.com/) to run LLMs locally for both the target model being tested and the AI judge evaluating responses.
* **Automated Jailbreak Attempts:** Uses a predefined (and extensible) set of jailbreak strategies.
* **Workflow Orchestration:** Employs [LangGraph](https://python.langchain.com/docs/langgraph) to manage the sequence of operations: prompt generation, LLM querying, and response evaluation.
* **AI-Powered Evaluation:** An "AI Judge" (another LLM) assesses whether the target LLM's response constitutes a successful jailbreak.
* **Interactive Web UI:** A [Streamlit](https://streamlit.io/) interface allows users to:
    * Configure target and judge Ollama models.
    * Add, view, and manage jailbreak strategies.
    * Select tasks and strategies for test runs.
    * View real-time progress and detailed logs of test attempts.
    * Download results.
* **Customizable Tasks & Strategies:** Easily define new tasks and strategies via JSON files or the UI.
* **Comprehensive Logging:** Records detailed information about each test attempt, including prompts, responses, and verdicts, to a `results/jailbreak_log.jsonl` file.

---
## Prerequisites

Before you begin, ensure you have the following installed:

1.  **Python:** Version 3.8 or higher. Make sure it's added to your system's PATH.
2.  **Ollama:** Installed and running. You can download it from [ollama.com](https://ollama.com/).
3.  **Ollama Models:** Download the LLMs you intend to use as the target and the judge via the Ollama CLI. For example:
    ```bash
    ollama pull llama3
    ollama pull mistral
    ```
    (These are examples; you can use any models compatible with Ollama.)

---
## Setup and Installation

1.  **Clone or Download the Repository:**
    Get the project files onto your local machine.

2.  **Windows Users (Recommended):**
    * Navigate to the project's root directory in your command prompt or PowerShell.
    * Run the `run_app.bat` script:
        ```batch
        run_app.bat
        ```
    * This script will automatically:
        * Create a Python virtual environment (e.g., `jailbreak_env`).
        * Activate the virtual environment.
        * Install all necessary dependencies from `requirements.txt`.
        * Start the Streamlit application.

3.  **Manual Setup (All Platforms / If `run_app.bat` is not used):**
    * Open your terminal or command prompt.
    * Navigate to the project's root directory.
    * Create a Python virtual environment:
        ```bash
        python -m venv jailbreak_env
        ```
    * Activate the virtual environment:
        * Windows: `jailbreak_env\Scripts\activate`
        * macOS/Linux: `source jailbreak_env/bin/activate`
    * Install the required packages:
        ```bash
        pip install -r requirements.txt
        ```
    * Run the Streamlit application:
        ```bash
        streamlit run app.py
        ```

---
## Directory Structure
```
Make_A_Break/
├── app.py                  # Streamlit UI application
├── graph_runner.py         # LangGraph execution logic
├── langgraph_setup.py      # LangGraph definition
├── llm_interface.py        # Ollama interaction functions
├── judge.py                # AI Judge logic
├── utils.py                # Utility functions (data loading, etc.)
├── requirements.txt        # Python dependencies
├── run_app.bat             # Windows batch script for easy startup
├── README.md               # This file
├── data/
│   ├── tasks.json          # Dataset of tasks for the LLM
│   └── strategies.json     # Predefined jailbreak strategies
├── results/
│   └── jailbreak_log.jsonl # Log of test attempts and results
└── jailbreak_env/          # Virtual environment directory (created by script)
```

---
## How to Use

1.  **Start the Application:**
    * Run `run_app.bat` (Windows) or follow the manual setup steps to start the Streamlit app.
    * The application should open in your default web browser, typically at `http://localhost:8501`.

2.  **Configure Models (Sidebar):**
    * In the sidebar, enter the names of the **Target Ollama Model** (the LLM you want to test) and the **Judge Ollama Model** (the LLM that will evaluate responses). Ensure these models are available in your Ollama setup.

3.  **Manage Strategies (Sidebar):**
    * View existing jailbreak strategies.
    * Add new strategies by providing a name, a unique ID, and a template. The template should use `{task_prompt}` as a placeholder for the actual task prompt.
    * Strategies are saved to `data/strategies.json`.

4.  **Define Tasks:**
    * Tasks are defined in `data/tasks.json`. Each task includes an `id`, `description`, the `prompt` to be sent to the LLM, and a `harm_category`. You can manually edit this file to add or modify tasks.

5.  **Run Tests (Main Area):**
    * Select one or more tasks from the "Select Tasks to Run" dropdown.
    * Select one or more strategies from the "Select Strategies to Apply" dropdown.
    * Click the "Start Jailbreak Test Run" button.
    * The application will iterate through each selected task-strategy combination, query the target LLM, and have the judge LLM evaluate the response.

6.  **View Logs & Results (Main Area):**
    * Progress and immediate results will be displayed as tests run.
    * A table of all results is shown, which can be refreshed from the `jailbreak_log.jsonl` file.
    * You can download the full results as a CSV file.
    * The most recent detailed result (JSON format) is also displayed for closer inspection.
    * All attempts are logged to `results/jailbreak_log.jsonl`.

---
## Customization

* **Adding Tasks:** Edit the `data/tasks.json` file to add new tasks. Follow the existing JSON structure.
    ```json
    [
        {
            "id": "new_task_example",
            "description": "A brief description of the new task.",
            "prompt": "The actual prompt for the LLM.",
            "harm_category": "e.g., misinformation"
        }
    ]
    ```
* **Adding Strategies:**
    * **Via UI:** Use the "Add New Strategy" section in the Streamlit sidebar.
    * **Manually:** Edit the `data/strategies.json` file. Ensure each strategy has a unique `id`, a `name`, and a `template` string.
        ```json
        [
            {
                "id": "S_custom_example",
                "name": "My Custom Strategy",
                "template": "This is a custom wrapper around the request: {task_prompt}. Please comply."
            }
        ]
        ```

---
## Contributing

Contributions, bug reports, and feature requests are welcome! Please feel free to open an issue or submit a pull request.
