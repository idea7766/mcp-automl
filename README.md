# MCP AutoML

MCP AutoML is a server that enables AI Agents to perform end-to-end machine learning workflows including data inspection, processing, model training. With MCP AutoML, AI Agents can perform more than a typical autoML framework. AI Agents can identify the target, setting baseline, and creating features by themselves.

MCP AutoML seperates tools and workflows, allowing you to create your own workflow.

## Features

- **Data Inspection**: Analyze datasets with comprehensive statistics, data types, and previews
- **SQL-based Data Processing**: Transform and engineer features using DuckDB SQL queries
- **AutoML Training**: Train classification and regression models with automatic model comparison using PyCaret
- **Prediction**: Make predictions using trained models
- **Multi-format Support**: Works with CSV, Parquet, and JSON files

## Usage

### Configure MCP Server

Add to your MCP client configuration (e.g., Claude Desktop, Gemini CLI, Cursor, Antigravity):

```json
{
  "mcpServers": {
    "mcp-automl": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/idea7766/mcp-automl", "mcp-automl"]
    }
  }
}
```

### Available Tools

| Tool | Description |
|------|-------------|
| `inspect_data` | Get comprehensive statistics and preview of a dataset |
| `query_data` | Execute DuckDB SQL queries on data files |
| `process_data` | Transform data using SQL and save to a new file |
| `train_classifier` | Train a classification model with AutoML |
| `train_regressor` | Train a regression model with AutoML |
| `predict` | Make predictions using a trained model |

## Agent Skill

MCP AutoML includes an **data science workflow skill** that guides AI agents through best practices for machine learning projects. This skill teaches agents to:

- Identify targets and establish baselines
- Perform exploratory data analysis
- Engineer domain-specific features
- Train and evaluate models systematically

### Installing the Skill

Copy the skill directory to your agent's skill folder:

```bash
# For Gemini Code Assist
cp -r skill/data-science-workflow ~/.gemini/skills/

# For Claude Code
cp -r skill/data-science-workflow ~/.claude/skills/

# For other agents, copy to their respective skill directories
```

The skill file is located at `skill/data-science-workflow/SKILL.md`.

## Configuration

Models and experiments are saved to `~/.mcp-automl/experiments/` by default.

## Dependencies

- [PyCaret](https://pycaret.org/) - AutoML library
- [DuckDB](https://duckdb.org/) - Fast SQL analytics
- [MCP](https://github.com/modelcontextprotocol/python-sdk) - Model Context Protocol SDK
