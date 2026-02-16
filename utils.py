# RedTools shared utilities
#
# Common functions extracted from across the codebase to eliminate duplication.
# Organised into sections:
#   1. Configuration
#   2. Text processing
#   3. Temporal framing
#   4. API response processing
#   5. BERTopic model initialisation

import re
import yaml
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sentence_splitter import SentenceSplitter
from nltk.corpus import stopwords
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load a YAML configuration file from the specified path."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


# ---------------------------------------------------------------------------
# 2. Text processing
# ---------------------------------------------------------------------------

_stopwords_cache = None


def _get_stopwords():
    """Return a cached set of English stopwords."""
    global _stopwords_cache
    if _stopwords_cache is None:
        _stopwords_cache = set(stopwords.words('english'))
    return _stopwords_cache


def pre_process(text: str) -> str:
    """Normalise a text string: strip HTML entities, URLs, mentions,
    non-alpha characters, and whitespace."""
    text = re.sub(r"&gt;", "", text)
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.replace("\n", "").replace("\t", "").strip()
    return text


def sentence_chunker(text: str) -> list:
    """Split text into sentences using SentenceSplitter."""
    splitter = SentenceSplitter(language="en")
    return splitter.split(text)


def expand_dataframe_with_sentences(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """Expand a DataFrame by splitting text into sentences, preprocessing
    each sentence, and retaining original row metadata."""
    df[text_column] = df[text_column].astype(str)
    df = df.dropna(subset=[text_column])
    df['sentences'] = df[text_column].apply(sentence_chunker)
    df_expanded = df.explode('sentences')

    df_expanded = df_expanded[df_expanded['sentences'].str.strip() != '']
    df_expanded['sentences'] = df_expanded['sentences'].apply(pre_process)
    df_expanded[text_column] = df_expanded['sentences']
    df_expanded = df_expanded.drop(columns=['sentences'])

    df_expanded = df_expanded[df_expanded[text_column].str.strip() != '']
    df_expanded = df_expanded.reset_index(drop=True)

    return df_expanded


def processed_text_column(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """Apply pre_process to every value in the specified text column."""
    df[text_column] = df[text_column].astype(str)
    df[text_column] = df[text_column].apply(pre_process)
    return df


def remove_stopwords(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """Remove English stopwords from the specified text column."""
    stop_words = _get_stopwords()
    df[text_column] = df[text_column].apply(
        lambda x: ' '.join([word for word in x.split() if word not in stop_words])
    )
    return df


# ---------------------------------------------------------------------------
# 3. Temporal framing
# ---------------------------------------------------------------------------

def get_frames(data: pd.DataFrame, date_column: str, timescale: str = 'week') -> list:
    """Split a DataFrame into time-based intervals.

    Supports timescales: 'hour', 'day', 'week', 'month', 'year'.
    Handles both Unix timestamp and datetime date columns.
    """
    if pd.api.types.is_numeric_dtype(data[date_column]):
        data[date_column] = pd.to_datetime(data[date_column], unit='s')
    else:
        data[date_column] = pd.to_datetime(data[date_column])

    start_time = data[date_column].min()
    end_time = data[date_column].max()

    freq_map = {
        'hour': 'h',
        'day': 'D',
        'week': 'W',
        'month': 'ME',
        'year': 'YE',
    }
    freq = freq_map.get(timescale)
    if freq is None:
        raise ValueError("Invalid timescale. Choose from 'hour', 'day', 'week', 'month', 'year'.")

    time_ranges = pd.date_range(start=start_time, end=end_time, freq=freq)

    frames = []
    time_pairs = list(zip(time_ranges[:-1], time_ranges[1:]))
    for start, end in tqdm(time_pairs, desc="Splitting into frames"):
        frame = data[(data[date_column] >= start) & (data[date_column] < end)].reset_index(drop=True)
        frames.append(frame)

    return frames


# ---------------------------------------------------------------------------
# 4. API response processing
# ---------------------------------------------------------------------------

class APIError(Exception):
    """Custom exception for API errors."""
    def __init__(self, status_code, detail):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"API Error {status_code}: {detail}")


def parse_unix_timestamp(timestamp):
    """Convert a Unix timestamp to a UTC datetime, or return None."""
    if timestamp is not None:
        return datetime.fromtimestamp(int(timestamp), tz=timezone.utc)
    return None


def process_subreddit_response(response):
    """Transform an API subreddit response into a DataFrame."""
    if isinstance(response, dict):
        subreddits = response.get('subreddits', [])
    elif isinstance(response, list):
        subreddits = response
    else:
        raise APIError(500, f"Invalid response format: expected a dictionary or list, got {type(response)}")

    if not subreddits:
        print("Warning: No subreddits found in the response")

    processed_data = []
    for subreddit in subreddits:
        if not isinstance(subreddit, dict):
            print(f"Warning: Unexpected subreddit format: {subreddit}")
            continue
        processed_subreddit = {
            'id': subreddit.get('id'),
            'display_name': subreddit.get('display_name'),
            'title': subreddit.get('title'),
            'description': subreddit.get('description'),
            'public_description': subreddit.get('public_description'),
            'created_utc': parse_unix_timestamp(subreddit.get('created_utc')),
            'subscribers': int(subreddit.get('subscribers', 0)),
            'over18': subreddit.get('over18'),
            'url': subreddit.get('url'),
            'banner_img': subreddit.get('banner_img'),
            'icon_img': subreddit.get('icon_img'),
            'community_icon': subreddit.get('community_icon'),
            'lang': subreddit.get('lang'),
        }
        processed_data.append(processed_subreddit)

    return pd.DataFrame(processed_data)


def process_submission_response(response):
    """Transform an API submission response into a DataFrame."""
    submissions = response.get('submissions', [])

    processed_data = []
    for submission in submissions:
        processed_submission = {
            'id': submission.get('id'),
            'title': submission.get('title'),
            'selftext': submission.get('selftext'),
            'author': submission.get('author'),
            'created_utc': parse_unix_timestamp(submission.get('created_utc')),
            'retrieved_utc': parse_unix_timestamp(submission.get('retrieved_utc')),
            'permalink': submission.get('permalink'),
            'url': submission.get('url'),
            'score': int(submission.get('score', 0)),
            'over_18': submission.get('over_18', False),
            'subreddit_id': submission.get('subreddit_id'),
            'subreddit': submission.get('subreddit'),
            'comment_count': int(submission.get('comment_count', 0)),
        }
        processed_data.append(processed_submission)

    return pd.DataFrame(processed_data)


def process_comment_response(response):
    """Transform an API comment response into a DataFrame."""
    comments = response.get('comments', [])

    processed_data = []
    for comment in comments:
        processed_comment = {
            'id': comment.get('id'),
            'author': comment.get('author'),
            'body': comment.get('body'),
            'created_utc': parse_unix_timestamp(comment.get('created_utc')),
            'submission_id': comment.get('submission_id'),
            'parent_id': comment.get('parent_id'),
            'score': int(comment.get('score', 0)),
            'subreddit_id': comment.get('subreddit_id'),
            'subreddit': comment.get('subreddit'),
            'permalink': comment.get('permalink'),
            'retrieved_on': parse_unix_timestamp(comment.get('retrieved_on')),
        }
        processed_data.append(processed_comment)

    return pd.DataFrame(processed_data)


def make_api_request(session, base_url, endpoint, api_key, params=None, method='GET', data=None):
    """Make an authenticated API request with standard error handling.

    Returns the parsed JSON response.
    """
    url = f"{base_url}/{endpoint}"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        if method == 'GET':
            response = session.get(url, params=params, headers=headers)
        elif method == 'POST':
            response = session.post(url, params=params, headers=headers, json=data)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        if response and response.status_code == 422:
            error_detail = response.json().get('detail', 'No detail provided')
            raise APIError(422, f"Unprocessable Entity: {error_detail}")
        elif response and 400 <= response.status_code < 500:
            raise APIError(response.status_code, f"Client Error: {response.text}")
        elif response and 500 <= response.status_code < 600:
            raise APIError(response.status_code, f"Server Error: {response.text}")
        else:
            raise APIError(http_err.response.status_code, f"HTTP Error: {str(http_err)}")
    except requests.exceptions.ConnectionError as conn_err:
        raise APIError(500, f"Connection Error: {str(conn_err)}")
    except requests.exceptions.Timeout as timeout_err:
        raise APIError(500, f"Timeout Error: {str(timeout_err)}")
    except requests.exceptions.RequestException as req_err:
        raise APIError(500, f"Request Error: {str(req_err)}")


# ---------------------------------------------------------------------------
# 5. BERTopic model initialisation
# ---------------------------------------------------------------------------

def create_umap_hdbscan_models(config: dict, hardware: str = 'CPU'):
    """Create UMAP and HDBSCAN model instances from config parameters.

    When hardware='GPU', uses cuml implementations; otherwise uses
    the CPU libraries (umap-learn and hdbscan).
    """
    umap_params = config['umap']
    hdbscan_params = config['hdbscan']

    if hardware == 'GPU':
        from cuml.manifold import UMAP
        from cuml.cluster import HDBSCAN
    else:
        from umap import UMAP
        from hdbscan import HDBSCAN

    umap_model = UMAP(
        n_components=umap_params['n_components'],
        n_neighbors=umap_params['n_neighbors'],
        min_dist=umap_params['min_dist'],
        random_state=umap_params['random_state'],
    )
    hdbscan_model = HDBSCAN(
        min_samples=hdbscan_params['min_samples'],
        gen_min_span_tree=hdbscan_params['gen_min_span_tree'],
        prediction_data=hdbscan_params['prediction_data'],
    )

    return umap_model, hdbscan_model


def create_bertopic_model(config: dict, hardware: str = 'CPU'):
    """Create a fully configured BERTopic model from config parameters."""
    from bertopic import BERTopic

    umap_model, hdbscan_model = create_umap_hdbscan_models(config, hardware)
    bertopic_params = config['bertopic']

    return BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=bertopic_params['calculate_probabilities'],
        verbose=bertopic_params['verbose'],
        min_topic_size=bertopic_params['min_topic_size'],
    )
