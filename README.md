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
| `config.yaml.example` | Template configuration file — copy this to `config.yaml` and fill in your credentials |

---

## Setup

### 1. Copy the example config

```bash
cp config.yaml.example config.yaml
```

`config.yaml` is excluded from version control via `.gitignore` so your credentials will never be accidentally committed.

### 2. Fill in your credentials

Open `config.yaml` and replace each placeholder with your real values:

```yaml
reddit:
  client_id: 'your_client_id'         # from https://www.reddit.com/prefs/apps
  client_secret: 'your_client_secret'
  redirect_uri: 'your_redirect_uri'
  user_agent: 'your_user_agent'

ausreddit:
  api_key: 'your_api_key'             # AusReddit collection API key

open_ai:
  api_key: 'your_openai_api_key'      # optional, only needed for OpenAI-backed tools

far_bot:
  google_api_key: 'your_google_api_key'         # Google AI Studio API key
  langsmith_tracing: 'true'
  langsmith_endpoint: 'https://api.smith.langchain.com'
  langsmith_api_key: 'your_langsmith_api_key'   # LangSmith API key
  langsmith_project: 'your_langsmith_project'   # LangSmith project name
```

The `umap`, `pca`, `tsvd`, `hdbscan`, `kmeans`, and `bertopic` sections contain hyperparameters that can be tuned — the defaults in `config.yaml.example` are a good starting point.

Set `hardware: CPU` if you do not have a GPU available.

---

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

### Output

- A feasibility report printed to the terminal (and optionally saved as a `.md` file)
- A bar chart of submission counts over time (`submission_frequency.png`)
- A line chart of ngram usage percentages over time (`ngram_volume.png`)
