import requests
import pandas as pd
import orjson
from typing import Iterator
import time
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone

from utils import (
    load_config,
    APIError,
    parse_unix_timestamp,
    process_subreddit_response,
    process_submission_response,
    process_comment_response,
    make_api_request,
)


config_path = 'config.yaml'

config = load_config(config_path)


@dataclass
class AusRedditComments:
    dataframe: pd.DataFrame
    embeddings: list[np.ndarray]
    length: int


class AusRedditData:  
    def __init__(self, base: str | None = None, config_path: str = 'config.yaml'):
        self.config = load_config(config_path)
        self.apikey = self.config['ausreddit']['api_key']
        if base is None:
            self.base_url = "https://ausreddit.digitalobservatory.net.au/api/v1"
        else:
            self.base_url = base + "/api/v1"
        self.session = requests.Session()

    def _make_endpoint_url(self, endpoint):
        return f"{self.base_url}/{endpoint}"

    def _make_request(self, endpoint, params=None, method="GET", data=None):
        return make_api_request(self.session, self.base_url, endpoint, self.apikey, params, method, data)

    def _parse_date_param(self, value, is_end: bool = False):
        """Convert a date value to a Unix timestamp integer.

        Accepts:
        - int/float: returned as-is, allowing fine-grained epoch control.
        - datetime: converted to a Unix timestamp.
        - str 'yyyy-mm-dd' or 'dd/mm/yyyy': converted to UTC start-of-day
          (00:00:00) for start dates, or UTC end-of-day (23:59:59) for end
          dates, so that a plain date covers the full calendar day.

        Raises ValueError for unrecognised string formats.
        """
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, datetime):
            return int(value.timestamp())
        if isinstance(value, str):
            for fmt in ('%Y-%m-%d', '%d/%m/%Y'):
                try:
                    dt = datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
                    if is_end:
                        dt = dt.replace(hour=23, minute=59, second=59)
                    return int(dt.timestamp())
                except ValueError:
                    continue
            raise ValueError(
                f"Unrecognised date format: {value!r}. "
                "Use 'yyyy-mm-dd', 'dd/mm/yyyy', or a Unix timestamp integer."
            )
        raise TypeError(f"Unsupported type for date parameter: {type(value).__name__}")

    def stream_comments(
        self, submissions: list[str], limit: int = 1000, embeddings: bool = False
    ) -> AusRedditComments:
        """
        This method streams data into a dataframe and optionally processes embeddings.

        Args:
        submissions (List[str]): List of submission IDs.
        limit (int): Limit on the number of comments to retrieve.
        embeddings (bool): Whether to retrieve and process embeddings.

        Returns:
        AusRedditComments: Object containing DataFrame of comments and list of embedding arrays.

        Raises:
        APIError: If an API error occurs during the request.
        """
        headers = {"Authorization": f"Bearer {self.apikey}"}
        params = {"submissions": submissions, "limit": limit, "embeddings": embeddings}
        endpoint_url = self._make_endpoint_url("comments/stream")
        column_types = {
            "id": "string",
            "author": "string",
            "body": "string",
            "created_utc": "datetime64[ns, UTC]",
            "submission_id": "string",
            "parent_id": "string",
            "subreddit_id": "string",
            "subreddit": "string",
            "permalink": "string",
            "retrieved_on": "datetime64[ns, UTC]",
        }

        df_list = []
        embedding_list = []
        chunk = []  # Chunks of 1000 comments
        total_processed = 0
        start_time = time.time()
        print("Downloading comments...")

        def stream_json() -> Iterator[dict]:
            try:
                with requests.get(
                    endpoint_url, headers=headers, params=params, stream=True
                ) as response:
                    if response.status_code != 200:
                        error_detail = (
                            response.text.strip() or "No error details provided"
                        )
                        raise APIError(response.status_code, error_detail)
                    for line in response.iter_lines(decode_unicode=True):
                        if line:
                            yield orjson.loads(line)
            except requests.RequestException as e:
                raise APIError(500, f"Network error occurred: {str(e)}")

        for item in stream_json():
            if embeddings and "embeddings" in item:
                embedding_array = np.array([int(bit) for bit in item['embeddings']], dtype=np.int8)
                embedding_list.append(embedding_array)
                del item[
                    "embeddings"
                ]  # Remove embedding from item to not include in DataFrame

            chunk.append(item)
            if len(chunk) >= 1000:
                df_chunk = pd.DataFrame(chunk).astype(column_types)
                df_list.append(df_chunk)
                total_processed += len(chunk)
                chunk = []

        # Process any remaining items
        if chunk:
            df_chunk = pd.DataFrame(chunk).astype(column_types)
            df_list.append(df_chunk)
            total_processed += len(chunk)

        end_time = time.time()
        elapsed_time = end_time - start_time
        comments_per_second = total_processed / elapsed_time

        print(
            f"{total_processed} comments downloaded in {elapsed_time:.2f} seconds ({comments_per_second:.2f} comments per second)"
        )

        # Combine all chunks into a single DataFrame
        df = (
            pd.concat(df_list, ignore_index=True)
            if df_list
            else pd.DataFrame(columns=column_types.keys()).astype(column_types)
        )

        return AusRedditComments(
            dataframe=df, embeddings=embedding_list, length=total_processed
        )

    def list_subreddits(self, meta=False):
        try:
            response = self._make_request('subreddits', params={'meta': meta})
            return self._process_subreddit_response(response)
        except APIError as e:
            print(f"Error listing subreddits: {e}")
            return pd.DataFrame()

    def _process_subreddit_response(self, response):
        return process_subreddit_response(response)

    def _parse_unix_timestamp(self, timestamp):
        return parse_unix_timestamp(timestamp)

    def get_submissions(self, subreddit_ids: str):
        """
        Fetches submissions for a given subreddit.

        Args:
            subreddit_id (str): The ID of the subreddit to fetch submissions from.

        Returns:
            pd.DataFrame: A DataFrame containing the submissions data. Returns an empty DataFrame if an error occurs.

        Raises:
            APIError: If there is an error making the request to the API.
        """
        try:
            response = self._make_request(f'submissions', params = {'subreddit_ids':subreddit_ids})
            return self._process_submission_response(response)
        except APIError as e:
            print(f"Error getting submissions: {e}")
            return pd.DataFrame()
        
    def _process_submission_response(self, response):
        return process_submission_response(response)

    def search_submissions(self, query, author=None,start=None, end=None, score_min=None, score_max=None, subreddit=None, subreddit_id=None, comments_min=None, comments_max=None, over18=None, method = 'keyword',search_in = 'all', limit = 1000, page = 1):
        """
        Search for submissions based on various criteria.

        Parameters:
        query (str): The search query string.
        author (str): The author of the submissions.
        method (str): The search method to use. Keyword or semantic.
        start (int | str): Start of the search range. Accepts a Unix timestamp
            (int/float) for fine-grained control, or a date string in
            'yyyy-mm-dd' or 'dd/mm/yyyy' format (interpreted as 00:00:00 UTC).
        end (int | str): End of the search range. Accepts a Unix timestamp
            (int/float) for fine-grained control, or a date string in
            'yyyy-mm-dd' or 'dd/mm/yyyy' format (interpreted as 23:59:59 UTC).
        score_min (int): The minimum score of the submissions.
        score_max (int): The maximum score of the submissions.
        subreddit (str): The subreddit to search in.
        subreddit_id (str): The ID of the subreddit to search in.
        limit (int): The maximum number of submissions to return (max 1000).
        search_in (str): The fields to search in.
        restricted (bool): search for NSFW or SFW submissions or both.
        comments_min (int): The minimum number of comments.
        comments_max (int): The maximum number of comments.
        page (int): The page number to retrieve (default 1).

        Returns:
        pd.DataFrame: A DataFrame containing the search results.

        Raises:
        APIError: If there is an error with the API request.
        """
        params = {
            'query': query,
            'author': author,
            'method': method,
            'start': self._parse_date_param(start, is_end=False),
            'end': self._parse_date_param(end, is_end=True),
            'score_min': score_min,
            'score_max': score_max,
            'subreddit': subreddit,
            'subreddit_id': subreddit_id,
            'limit': limit,
            'search_in': search_in,
            'over18': over18,
            'comments_min': comments_min,
            'comments_max': comments_max,
            'page': page,
        }
        try:
            response = self._make_request('submissions', params=params)
            return self._process_submission_response(response)
        except APIError as e:
            print(f"Error searching submissions: {e}")
            return pd.DataFrame()

    def get_all_submissions(self, query, author=None, start=None, end=None, score_min=None, score_max=None, subreddit=None, subreddit_id=None, comments_min=None, comments_max=None, over18=None, method='keyword', search_in='all', limit=1000):
        """
        Fetch all pages of submission search results automatically.

        Accepts the same parameters as search_submissions() (except page).
        Iterates through pages until the API reports no further results.

        Returns:
        pd.DataFrame: A DataFrame containing all results across all pages.
        """
        base_params = {
            'query': query,
            'author': author,
            'method': method,
            'start': self._parse_date_param(start, is_end=False),
            'end': self._parse_date_param(end, is_end=True),
            'score_min': score_min,
            'score_max': score_max,
            'subreddit': subreddit,
            'subreddit_id': subreddit_id,
            'limit': limit,
            'search_in': search_in,
            'over18': over18,
            'comments_min': comments_min,
            'comments_max': comments_max,
        }
        all_pages = []
        page = 1
        while True:
            try:
                response = self._make_request('submissions', params={**base_params, 'page': page})
                df = self._process_submission_response(response)
                if not df.empty:
                    all_pages.append(df)
                if not response.get('next', False):
                    break
                page += 1
            except APIError as e:
                print(f"Error fetching submissions page {page}: {e}")
                break
        return pd.concat(all_pages, ignore_index=True) if all_pages else pd.DataFrame()
        
    def _process_comment_response(self, response):
        return process_comment_response(response)
        
    def search_comments(self, query, author= None, start= None, end= None, score_min= None, score_max= None, subreddit= None, subreddit_id = None, over18 = None, method = 'keyword',search_in = 'all', limit = 1000, page = 1):
        """
        Search for comments based on various criteria.
        Parameters:
        -----------
        query : str
            The search query string.
        author : str
            The author of the comments.
        method : str
            The search method to use.
        start : int or str
            Start of the search range. Accepts a Unix timestamp (int/float)
            for fine-grained control, or a date string in 'yyyy-mm-dd' or
            'dd/mm/yyyy' format (interpreted as 00:00:00 UTC).
        end : int or str
            End of the search range. Accepts a Unix timestamp (int/float)
            for fine-grained control, or a date string in 'yyyy-mm-dd' or
            'dd/mm/yyyy' format (interpreted as 23:59:59 UTC).
        score_min : int
            The minimum score of the comments.
        score_max : int
            The maximum score of the comments.
        subreddit : str
            The subreddit to search in.
        subreddit_id : str
            The ID of the subreddit to search in.
        limit : int
            The maximum number of comments to return (max 1000).
        search_in : str
            The fields to search in.
        restricted : bool
            Whether to restrict the search to certain criteria. NSFW or SFW comments or both.
        page : int
            The page number to retrieve (default 1).
        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the search results.
        Raises:
        -------
        APIError
            If there is an error during the API request.
        """
        params = {
            'query': query,
            'author': author,
            'method': method,
            'start': self._parse_date_param(start, is_end=False),
            'end': self._parse_date_param(end, is_end=True),
            'score_min': score_min,
            'score_max': score_max,
            'subreddit': subreddit,
            'subreddit_id': subreddit_id,
            'limit': limit,
            'search_in': search_in,
            'over18': over18,
            'page': page,
        }

        try:
            response = self._make_request('comments', params=params)
            return self._process_comment_response(response)
        except APIError as e:
            print(f"Error searching comments: {e}")
            return pd.DataFrame()

    def get_all_comments(self, query, author=None, start=None, end=None, score_min=None, score_max=None, subreddit=None, subreddit_id=None, over18=None, method='keyword', search_in='all', limit=1000):
        """
        Fetch all pages of comment search results automatically.

        Accepts the same parameters as search_comments() (except page).
        Iterates through pages until the API reports no further results.

        Returns:
        pd.DataFrame: A DataFrame containing all results across all pages.
        """
        base_params = {
            'query': query,
            'author': author,
            'method': method,
            'start': self._parse_date_param(start, is_end=False),
            'end': self._parse_date_param(end, is_end=True),
            'score_min': score_min,
            'score_max': score_max,
            'subreddit': subreddit,
            'subreddit_id': subreddit_id,
            'limit': limit,
            'search_in': search_in,
            'over18': over18,
        }
        all_pages = []
        page = 1
        while True:
            try:
                response = self._make_request('comments', params={**base_params, 'page': page})
                df = self._process_comment_response(response)
                if not df.empty:
                    all_pages.append(df)
                if not response.get('next', False):
                    break
                page += 1
            except APIError as e:
                print(f"Error fetching comments page {page}: {e}")
                break
        return pd.concat(all_pages, ignore_index=True) if all_pages else pd.DataFrame()
        
    def get_comments(self, submission_ids):
        try:
            response = self._make_request(f'comments', params ={'submission_ids': submission_ids})
            return self._process_comment_response(response)
        except APIError as e:
            print(f"Error getting comments: {e}")
            return pd.DataFrame()

    def get_submission_aggregates(self, query, start=None, end=None, period='week'):
        """
        Fetch submission frequency aggregated into time bins.

        Uses GET /aggregates/submissions/.

        Parameters:
        -----------
        query : str
            Search text to filter submissions.
        start : int or str, optional
            Start of the range. Accepts a Unix timestamp (int/float) or a date
            string in 'yyyy-mm-dd' / 'dd/mm/yyyy' format (interpreted as
            00:00:00 UTC). Defaults to the earliest record in the DB.
        end : int or str, optional
            End of the range. Accepts a Unix timestamp (int/float) or a date
            string in 'yyyy-mm-dd' / 'dd/mm/yyyy' format (interpreted as
            23:59:59 UTC). Defaults to the latest record in the DB.
        period : str, optional
            Bin size: 'day', 'week', 'month', or 'year'. Default 'week'.

        Returns:
        --------
        pd.DataFrame
            Columns: start (datetime, UTC), end (datetime, UTC), frequency (int).
        """
        params = {
            'query': query,
            'start': self._parse_date_param(start, is_end=False),
            'end': self._parse_date_param(end, is_end=True),
            'period': period,
        }
        try:
            response = self._make_request('aggregates/submissions', params=params)
            results = response.get('results', [])
            df = pd.DataFrame(results)
            if not df.empty:
                df['start'] = pd.to_datetime(df['start'], unit='s', utc=True)
                df['end'] = pd.to_datetime(df['end'], unit='s', utc=True)
            return df
        except APIError as e:
            print(f"Error fetching submission aggregates: {e}")
            return pd.DataFrame()

    def _parse_year_month(self, value):
        """Parse a date value and return a (year, month) tuple.

        Accepts:
        - str 'yyyy-mm-dd' or 'dd/mm/yyyy': year and month extracted.
        - str 'yyyy-mm': year and month extracted directly.
        - int: treated as a year (month returned as None).

        Raises ValueError for unrecognised formats.
        """
        if isinstance(value, int):
            return value, None
        if isinstance(value, str):
            for fmt in ('%Y-%m-%d', '%d/%m/%Y'):
                try:
                    dt = datetime.strptime(value, fmt)
                    return dt.year, dt.month
                except ValueError:
                    continue
            try:
                dt = datetime.strptime(value, '%Y-%m')
                return dt.year, dt.month
            except ValueError:
                pass
            raise ValueError(
                f"Unrecognised date format: {value!r}. "
                "Use 'yyyy-mm-dd', 'dd/mm/yyyy', 'yyyy-mm', or an integer year."
            )
        raise TypeError(f"Unsupported type for date parameter: {type(value).__name__}")

    def get_ngrams(self, queries, start=None, end=None):
        """
        Fetch ngram frequency timelines.

        Uses POST /aggregates/ngrams/.

        Each query is 1–3 words. The first (and only the first) query may
        contain a '*' wildcard as a whole word, in which case the API expands
        it to the top 5 matching ngrams and returns them as separate series.

        Parameters:
        -----------
        queries : list[str]
            One or more ngram strings to look up.
        start : str or int, optional
            Start of the range. Accepts 'yyyy-mm-dd', 'dd/mm/yyyy', 'yyyy-mm',
            or an integer year. Year and month are extracted and sent to the API.
        end : str or int, optional
            End of the range. Same formats as start.

        Returns:
        --------
        pd.DataFrame
            Index: period labels (e.g. '2021-01'). Columns: one per query (or
            expanded wildcard ngram). Values are percentage of total comments
            for that month, rounded to 4 decimal places.
        """
        body = {'queries': queries}
        if start is not None:
            start_year, start_month = self._parse_year_month(start)
            body['start_year'] = start_year
            if start_month is not None:
                body['start_month'] = start_month
        if end is not None:
            end_year, end_month = self._parse_year_month(end)
            body['end_year'] = end_year
            if end_month is not None:
                body['end_month'] = end_month

        try:
            response = self._make_request('aggregates/ngrams', method='POST', data=body)
            timeline = response.get('timeline', [])
            values = response.get('values', {})
            df = pd.DataFrame(values, index=timeline)
            df.index.name = 'period'
            return df
        except APIError as e:
            print(f"Error fetching ngrams: {e}")
            return pd.DataFrame()