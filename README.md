# Financial RAG System

A local RAG (Retrieval-Augmented Generation) system for querying your personal financial PDF documents using Claude AI.

## 🎯 What This Does

This system lets you ask natural language questions about your financial documents:

- "What were my grocery expenses in March 2024?"
- "How much did I earn last quarter?"
- "Compare my spending between January and February"
- "What are my biggest expense categories this year?"

It works by:

1. **Indexing** your PDF financial documents into a local vector database
2. **Finding** relevant document chunks when you ask a question
3. **Generating** accurate answers using Claude with your document context

## � Table of Contents

- [What This Does](#-what-this-does)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
  - [Step 1: Install Dependencies](#step-1-install-dependencies)
  - [Step 2: VSCode IDE Setup](#step-2-vscode-ide-setup)
  - [Step 3: Configure Environment](#step-3-configure-environment)
  - [Step 4: Run the Quick Start Example](#step-4-run-the-quick-start-example)
  - [Repopulating the Database and Running MCP Server](#repopulating-the-database-and-running-mcp-server)
- [Code Quality with Ruff](#-code-quality-with-ruff)
- [Unit Testing](#-unit-testing)
- [Claude Desktop Integration (MCP)](#️-claude-desktop-integration-mcp)
- [Auto-Sync New Documents](#-auto-sync-new-documents)
- [Running Commands](#-running-commands)
- [Available MCP Tools](#-available-mcp-tools)
- [Configuration Options](#-configuration-options)
- [Cost Breakdown](#-cost-breakdown)
- [Security Notes](#-security-notes)
- [Troubleshooting](#-troubleshooting)
- [Learning Resources](#-learning-resources)
- [License](#-license)

## �📁 Project Structure

```
financial-rag/
├── src/
│   └── financial_rag/
│       ├── __init__.py         # Package initialization
│       ├── rag_pipeline.py     # Core RAG system
│       ├── mcp_server.py       # MCP server for Claude Desktop
│       └── file_watcher.py     # Auto-sync when new files are added
├── .vscode/
│   ├── settings.json           # Configuration for VSCode
│   └── extensions.json         # Recommended extensions
├── tests/
│   ├── __init__.py             # Test package
│   ├── conftest.py             # Pytest configuration and fixtures
│   └── test_rag_pipeline.py    # RAG pipeline tests
├── pyproject.toml              # Dependencies and project config
├── .env.example                # Environment variables template
├── .env                        # Your actual config (create this)
├── .pre-commit-config.yaml     # Pre-commit hooks configuration
├── chroma_db/                  # Vector database (auto-created)
└── README.md                   # This file
```

## 🚀 Quick Start

### Step 1: Install Dependencies

Using uv (recommended - you already have this!):

```bash
# Navigate to the project directory
cd financial-rag

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Mac/Linux
# .venv\Scripts\activate   # On Windows

# Install all dependencies
uv pip install -e .
```

Or using pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Step 2: VSCode IDE Setup

#### Pylance Import Resolution

After Step 1, Pylance in VSCode should recognize imports like `from financial_rag.rag_pipeline import FinancialRag`.

If you see "FinancialRag is unknown import symbol" error:

1. The `uv pip install -e .` from Step 1 installs the package in **editable mode**, which makes it discoverable to Pylance
2. Reload VS Code: `Cmd+Shift+P` → "Reload Window"
3. Or clear Pylance cache: `Cmd+Shift+P` → "Pylance: Clear Cache"

This only needs to be done once after the initial setup.

#### (Optional) VSCode Formatting with Ruff

For automatic code formatting and linting on save:

1. Install the [Ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) for VSCode
   - Or accept the automatic recommendation prompt if you open the project in VSCode

2. The project includes `.vscode/settings.json` that automatically enables:
   - Format on save with ruff
   - Auto-fix linting issues on save
   - Import organization

### Step 3: Configure Environment

```bash
# Copy the example env file
cp .env.example .env

# Edit .env with your Anthropic API key
# Get your key from: https://console.anthropic.com/
```

Edit `.env`:

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Step 4: Run the Quick Start Example

The `example.py` script guides you through the entire setup process:

```bash
# Recommended: Using uv run (automatically uses virtual environment)
uv run python example.py

# Alternative: Activate venv first, then run
source .venv/bin/activate
python example.py
```

This script will:

1. ✅ Verify your API key is configured
2. 📂 Auto-detect your financial documents folder
3. 📥 Offer to ingest your PDFs
4. 💡 Run demo queries to show how it works
5. 📋 Explain all available usage options

The script handles all the details—just follow the prompts!

**Note:** You need to run this within the virtual environment. Using `uv run` is the easiest approach as it automatically handles the virtual environment without requiring manual activation.

### Repopulating the Database and Running MCP Server

If you want to clear all indexed documents and repopulate the database before running the MCP server, follow this workflow:

#### Step 1: Clear the Existing Database

```bash
# Option 1: Delete the database directory (simplest)
rm -rf ./chroma_db

# Option 2: Clear programmatically
uv run python -c "
from financial_rag.rag_pipeline import FinancialRag
rag = FinancialRag()
rag.clear_database()
print('✅ Database cleared')
"
```

#### Step 2: Repopulate with Your Documents

```bash
# Option 1: Use the example script (interactive and guided)
uv run python example.py
# Follow the prompts to ingest your PDFs

# Option 2: Ingest programmatically
uv run python -c "
from financial_rag.rag_pipeline import FinancialRag
from pathlib import Path

rag = FinancialRag()
docs_path = Path('~/OneDrive/Documents/Finance/Financial History').expanduser()
rag.ingest_documents(docs_path)
print('✅ Documents ingested')
"

# Option 3: Ingest from a custom path
uv run python -c "
from financial_rag.rag_pipeline import FinancialRag
from pathlib import Path

rag = FinancialRag()
docs_path = Path('/your/custom/path/to/documents').expanduser()
rag.ingest_documents(docs_path)
print('✅ Documents ingested')
"
```

#### Step 3: Run the MCP Server

See the [Running the MCP Server](#running-the-mcp-server) section below for detailed instructions.

#### Complete One-Liner Workflow

```bash
# Clear database, repopulate, and run MCP server in sequence
rm -rf ./chroma_db && \
uv run python -c "
from financial_rag.rag_pipeline import FinancialRag
from pathlib import Path
rag = FinancialRag()
docs_path = Path('~/OneDrive/Documents/Finance/Financial History').expanduser()
rag.ingest_documents(docs_path)
print('✅ Database repopulated')
" && \
uv run python -m financial_rag.mcp_server
```

**Note:** The MCP server will continue running until you stop it (Ctrl+C). Make sure to restart Claude Desktop after stopping/starting the MCP server to ensure it connects properly.

## � Code Quality with Ruff

This project uses [Ruff](https://docs.astral.sh/ruff/) for fast Python formatting and linting.

### VSCode Integration (Automatic)

Once the Ruff extension is installed, formatting and linting happens automatically on save:

- Fixes import organization
- Formats code according to configured style
- Fixes common linting issues

### Manual Formatting

```bash
# Format all Python files
ruff format .

# Check and fix linting issues
ruff check . --fix

# Check without fixing
ruff check .
```

### Pre-commit Hooks (Optional)

To automatically run ruff checks before each commit, set up pre-commit hooks:

1. Install pre-commit:

```bash
pip install pre-commit
```

2. Install the git hooks:

```bash
pre-commit install
```

Now ruff will automatically format and lint your code before each commit. To run manually:

```bash
# Run all hooks on all files
pre-commit run --all-files

# Run specific hook
pre-commit run ruff --all-files
```

## 🧪 Unit Testing

This project uses [pytest](https://docs.pytest.org/) for unit testing.

### Running Tests

```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run a specific test file
pytest tests/test_rag_pipeline.py

# Run a specific test
pytest tests/test_rag_pipeline.py::TestFinancialRagBasics::test_rag_import

# Run tests with coverage report
pytest --cov=. --cov-report=html
```

### Test Structure

Tests are located in the `tests/` directory:

- `test_rag_pipeline.py` - Core RAG pipeline tests
- `conftest.py` - Shared pytest fixtures and configuration

### Writing Tests

Example test structure:

```python
import pytest

class TestMyFeature:
    """Test suite for a feature."""

    def test_basic_functionality(self):
        """Test basic functionality."""
        from financial_rag.rag_pipeline import FinancialRag
        rag = FinancialRag()
        assert rag is not None

    def test_with_fixture(self, sample_query):
        """Test using a fixture."""
        # Use the sample_query fixture
        assert len(sample_query) > 0
```

## �🖥️ Claude Desktop Integration (MCP)

To use this directly from Claude Desktop:

### Option 1: Using Desktop Extensions (Easiest)

Claude Desktop now supports one-click MCP installations. Check Settings > Extensions for available options.

### Option 2: Manual Configuration

1. Find your Claude Desktop config file:
   - **Mac**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

2. Get your absolute paths:

```bash
# From the financial-rag directory, get the Python executable path
which python  # After activating the venv (e.g., /Users/you/code/financial-rag/.venv/bin/python)

# Get the project path
pwd  # (e.g., /Users/you/code/financial-rag)
```

3. Add this configuration to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "financial-rag": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": ["-m", "financial_rag.mcp_server"],
      "env": {
        "ANTHROPIC_API_KEY": "your-key-here"
      }
    }
  }
}
```

**Example** (replace with your actual paths):

```json
{
  "mcpServers": {
    "financial-rag": {
      "command": "/Users/belden/code/financial-rag/.venv/bin/python",
      "args": ["-m", "financial_rag.mcp_server"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-your-key-here"
      }
    }
  }
}
```

4. Restart Claude Desktop

5. You should see the financial RAG tools available in Claude Desktop!

## 📂 Auto-Sync New Documents

To automatically index new PDFs when they're added to your OneDrive:

```bash
# Start the file watcher
python -m financial_rag.file_watcher

# Or specify a custom path
python -m financial_rag.file_watcher "~/OneDrive/Documents/Finance/Financial History"
```

Or use the installed command:

```bash
# After running 'uv pip install -e .'
financial-watcher

# Or with a custom path
financial-watcher "~/OneDrive/Documents/Finance/Financial History"
```

This runs in the background and watches for:

- New PDF files being added
- Existing PDF files being modified

When detected, they're automatically indexed.

**Tip**: You can run this as a background service or add it to your startup items.

## � Running Commands

After installing with `uv pip install -e .`, you can use these commands.

**Note:** All commands should be run using `uv run` to automatically use the virtual environment, or activate the virtual environment first with `source .venv/bin/activate`.

### Running the MCP Server

```bash
# Using uv run (recommended - no activation needed)
uv run python -m financial_rag.mcp_server

# Or directly if venv is activated
python -m financial_rag.mcp_server

# Or using the installed command alias
financial-mcp
```

The MCP server will start and listen for connections from Claude Desktop.

### Running the CLI

```bash
# Using uv run (recommended - no activation needed)
uv run python -m financial_rag.rag_pipeline

# Or directly if venv is activated
python -m financial_rag.rag_pipeline

# Or using the installed command alias
financial-rag
```

Interactive CLI for querying documents.

### Running the File Watcher

```bash
# Using uv run (recommended - no activation needed)
uv run python -m financial_rag.file_watcher

# Or with a custom path
uv run python -m financial_rag.file_watcher "~/OneDrive/Documents/Finance/Financial History"

# Or directly if venv is activated
python -m financial_rag.file_watcher "/your/custom/path"

# Or using the installed command alias
financial-watcher
financial-watcher "/your/custom/path"
```

## �📊 Available MCP Tools

When connected to Claude Desktop, these tools are available:

| Tool                      | Description                                |
| ------------------------- | ------------------------------------------ |
| `query_finances`          | Ask any question about your documents      |
| `get_monthly_expenses`    | Get expense breakdown for a specific month |
| `get_monthly_income`      | Get income for a specific month            |
| `compare_income_expenses` | Compare income vs expenses                 |
| `search_transactions`     | Search for specific transactions           |
| `get_expense_trends`      | Analyze spending trends over time          |
| `get_financial_summary`   | Get yearly financial summary               |
| `get_database_stats`      | Check how many documents are indexed       |

## 🔧 Configuration Options

### Environment Variables

| Variable              | Default                                          | Description                        |
| --------------------- | ------------------------------------------------ | ---------------------------------- |
| `ANTHROPIC_API_KEY`   | (required)                                       | Your Anthropic API key             |
| `FINANCIAL_DOCS_PATH` | `~/OneDrive/Documents/Finance/Financial History` | Path to your documents             |
| `CHROMA_DB_PATH`      | `./chroma_db`                                    | Where to store the vector database |
| `CLAUDE_MODEL`        | `claude-sonnet-4-20250514`                       | Which Claude model to use          |
| `EMBEDDING_MODEL`     | `sentence-transformers/all-MiniLM-L6-v2`         | Local embedding model              |

### Expected Document Format

The system works best with PDFs named like:

- `Family Expenses 2024-01-01.pdf`
- `Income Statement 2024-01-01.pdf`

Organized in folders like:

```
Financial History/
├── Financial History (2023)/
│   ├── Family Expenses 2023-01-01.pdf
│   ├── Income Statement 2023-01-01.pdf
│   └── ...
├── Financial History (2024)/
│   ├── Family Expenses 2024-01-01.pdf
│   └── ...
```

But it will work with any PDF structure - the naming just helps with metadata extraction.

## 💰 Cost Breakdown

This local setup is **very cheap**:

| Component  | Cost                                             |
| ---------- | ------------------------------------------------ |
| ChromaDB   | **Free** (runs locally)                          |
| Embeddings | **Free** (HuggingFace, runs locally)             |
| Claude API | **~$0.003-0.015 per query** (depending on model) |

For typical personal use (20-50 queries/month), expect **< $1/month**.

## 🔒 Security Notes

- All your financial data stays **on your local machine**
- The vector database is stored locally in `./chroma_db`
- Only the query context is sent to Claude's API (not your entire database)
- Your API key should be kept secret (never commit `.env` to git)

## 🐛 Troubleshooting

### "FinancialRag is unknown import symbol" (Pylance error in VSCode)

This happens when the package isn't installed in the virtual environment that Pylance is using.

**Solution:**

1. Ensure you ran `uv pip install -e .` in Step 1
2. Reload VS Code: `Cmd+Shift+P` → "Reload Window"
3. Or clear Pylance cache: `Cmd+Shift+P` → "Pylance: Clear Cache"

The editable install (`-e` flag) makes your local package discoverable to the IDE without needing to reinstall when you make changes.

### "ANTHROPIC_API_KEY not found"

Make sure you've created `.env` with your API key:

```bash
cp .env.example .env
# Then edit .env with your key
```

### "No documents found"

Check that your path is correct and contains PDF files:

```python
from pathlib import Path
path = Path("~/OneDrive/Documents/Finance/Financial History").expanduser()
print(f"Path exists: {path.exists()}")
print(f"PDF files: {list(path.rglob('*.pdf'))}")
```

### Slow first query

The first query takes longer because:

1. Embedding model needs to download (~90MB, one-time)
2. Model needs to load into memory

Subsequent queries are much faster.

### MCP server not appearing in Claude Desktop

1. Verify paths in `claude_desktop_config.json` are absolute paths
2. Make sure the virtual environment is activated in the command
3. Restart Claude Desktop completely (quit and reopen)
4. Check the Extensions settings panel for any error logs

## 📚 Learning Resources

- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/pdf_qa/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [MCP Protocol Documentation](https://modelcontextprotocol.io/)
- [Claude API Documentation](https://docs.anthropic.com/)

## 📝 License

MIT License - feel free to modify and use for your personal finances!
