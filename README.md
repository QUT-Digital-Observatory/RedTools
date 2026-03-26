# RedTools

A collection of tools for analysing Reddit data from the AusReddit collection, designed for deployment on Nectar BinderHub.

## Contents

| File | Description |
|------|-------------|
| `api.py` | API wrapper for the AusReddit collection |
| `far_bot.py` | Feasibility assessment bot — evaluates whether a topic has sufficient data for research |
| `emo_intensity_over_time.py` | NRC emotion intensity analysis over time |
| `LDA_over_time.py` | LDA topic modelling over time |
| `NLP_over_time.py` | Basic NLP analysis |
| `topic_window.py` | BERTopic and BERTopic with time windows |
| `config.yaml` | Configuration file for options, hyperparameters, and API keys (see setup below) |

## Setup

Copy `config.yaml` and fill in your credentials. The file is excluded from version control via `.gitignore` to prevent accidental secret exposure.

```yaml
ausreddit:
  api_key: 'your_api_key'

far_bot:
  google_api_key: 'your_google_api_key'
  langsmith_tracing: 'true'
  langsmith_endpoint: 'https://api.smith.langchain.com'
  langsmith_api_key: 'your_langsmith_api_key'
  langsmith_project: 'your_langsmith_project'
```

## Feasibility Assessment Bot (`far_bot.py`)

Assesses whether a topic has enough data in the AusReddit collection to be worth studying. Given a query and date range it retrieves submission counts and ngram frequencies, generates charts, and produces a short report covering:

- **Occurrence** — is the topic present, and when does it first/last appear?
- **Frequency** — how many submissions mention it over time?
- **Volume** — what proportion of total comments mention it?

### Usage

**Command line:**
```bash
python far_bot.py "bluey" --start 2024-01-01 --end 2025-01-01 --save
```

**As a module:**
```python
from far_bot import run
run("bluey", start="2024-01-01", end="2025-01-01", save=True)
```

The `--save` / `save=True` flag writes the report (`.md`) and charts (`.png`) to files named after the topic.

### Date formats

`--start` and `--end` accept `yyyy-mm-dd` or `dd/mm/yyyy`.
