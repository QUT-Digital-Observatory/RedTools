# RedTools

A collection of tools for analysing Reddit data from the AusReddit collection, designed for deployment on Nectar BinderHub.

## Contents

| File | Description |
|------|-------------|
| `api.py` | API wrapper for the AusReddit collection |
| `far_bot.py` | Feasibility assessment bot — evaluates whether a topic has sufficient data for research |
| `reddit_topic_trees.py` | Builds directed conversation graphs from Reddit submissions and comments |
| `ausreddit_metrics.py` | Computes per-submission conversation metrics from a conversation graph |
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

## Conversation Trees (`reddit_topic_trees.py` + `ausreddit_metrics.py`)

`Reddit_trees` collects Reddit data via the PRAW API and builds directed conversation
graphs where each node is a submission or comment, and each edge represents a
reply relationship.

`AusredditMetrics` takes one of those graphs and returns a DataFrame of per-submission
structural and time-based metrics.

### Building a conversation graph

```python
from reddit_topic_trees import Reddit_trees
from ausreddit_metrics import AusredditMetrics

trees = Reddit_trees()

# Collect data
submissions_df = trees.search_subreddit("housing affordability", subreddit="australia")
comments_df = trees.fetch_comments(submissions_df['id'].tolist())

# Build graph — submissions_df is required so each submission becomes the root node
G, adj = trees.tree_graph_and_adj_list(comments_df, submissions_df)
```

Each connected component in `G` corresponds to exactly one submission and all of
its comments. The submission node (in-degree 0) is the root of each tree.

### Computing metrics

```python
metrics = AusredditMetrics()
df = metrics.analyze_conversation_graphs(G)
print(df)
```

The returned DataFrame is indexed by submission ID and includes:

| Column | Description |
|--------|-------------|
| `num_comments` | Number of comment nodes (excludes the submission root) |
| `num_nodes` | Total nodes including the submission root |
| `num_edges` | Number of reply edges |
| `longest_path_length` | Depth of the deepest reply chain |
| `average_path_length` | Mean depth across all nodes |
| `num_branches` | Nodes where more than one reply was made |
| `num_endpoints` | Leaf nodes (comments with no replies) |
| `total_duration` | Time from submission to last comment (HH:MM:SS) |
| `shortest_response_time` | Fastest reply in the thread (HH:MM:SS) |
| `longest_response_time` | Slowest reply in the thread (HH:MM:SS) |
| `average_response_time` | Mean reply time across all edges (HH:MM:SS) |

### Column name overrides

`tree_graph_and_adj_list` accepts keyword arguments to remap column names for
non-Reddit data schemas:

```python
G, adj = trees.tree_graph_and_adj_list(
    comments_df,
    submissions_df,
    id_col='commentId',
    author_col='username',
    body_col='text',
    link_id_col='threadId',
    parent_id_col='responseTo',
    time_col='date',
    time_is_utc=False,
    submission_title_col='headline',
    submission_body_col='content',
)
```

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
