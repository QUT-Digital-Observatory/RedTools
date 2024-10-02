import requests
import pandas as pd
import orjson
from typing import Iterator
import time
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone


class APIError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"API Error {status_code}: {detail}")


@dataclass
class AusRedditComments:
    dataframe: pd.DataFrame
    embeddings: list[np.ndarray]
    length: int


class AusRedditData:
    def __init__(self, apikey: str, base: str | None = None):
        self.apikey = apikey
        if base is None:
            self.base_url = "https://ausreddit.digitalobservatory.net.au/api/v1"
        else:
            self.base_url = base + "/api/v1"
        self.session = requests.Session()

    def _make_endpoint_url(self, endpoint):
        return f"{self.base_url}/{endpoint}"

    def _make_request(self, endpoint, params=None, method="GET", data=None):
        url = f"{self.base_url}/{endpoint}"
        headers = {"Authorization": f"Bearer {self.apikey}"}
        try:
            if method == "GET":
                response = self.session.get(url, params=params, headers=headers)
            elif method == "POST":
                response = self.session.post(
                    url, params=params, headers=headers, json=data
                )
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            # Here, we're sure that 'response' exists because HTTPError is only raised after a response is received
            if response and response.status_code == 422:  # type: requests.Response
                error_detail = response.json().get("detail", "No detail provided")
                raise APIError(422, f"Unprocessable Entity: {error_detail}")
            elif response and 400 <= response.status_code < 500:
                raise APIError(response.status_code, f"Client Error: {response.text}")
            elif response and 500 <= response.status_code < 600:
                raise APIError(response.status_code, f"Server Error: {response.text}")
            else:
                raise APIError(
                    http_err.response.status_code, f"HTTP Error: {str(http_err)}"
                )
        except requests.exceptions.ConnectionError as conn_err:
            raise APIError(500, f"Connection Error: {str(conn_err)}")
        except requests.exceptions.Timeout as timeout_err:
            raise APIError(500, f"Timeout Error: {str(timeout_err)}")
        except requests.exceptions.RequestException as req_err:
            raise APIError(500, f"Request Error: {str(req_err)}")

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
        # Assuming the response is a list of subreddit dictionaries
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
                'created_utc': self._parse_unix_timestamp(subreddit.get('created_utc')),
                'subscribers': int(subreddit.get('subscribers', 0)),
                'over18': subreddit.get('over18'),
                'url': subreddit.get('url'),
                'banner_img': subreddit.get('banner_img'),
                'icon_img': subreddit.get('icon_img'),
                'community_icon': subreddit.get('community_icon'),
                'lang': subreddit.get('lang')
            }
            processed_data.append(processed_subreddit)
        
        return pd.DataFrame(processed_data)

    def _parse_unix_timestamp(self, timestamp):
        """
        Parses a Unix timestamp and returns a datetime object in UTC.

        Args:
            timestamp (int or None): The Unix timestamp to parse. If None, returns None.

        Returns:
            datetime.datetime or None: A datetime object representing the given timestamp in UTC, or None if the input is None.
        """
        if timestamp is not None:
            return datetime.fromtimestamp(int(timestamp), tz=timezone.utc)
        return None

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
        """
        Processes a response containing Reddit submissions and converts it into a DataFrame.
        Args:
            response (dict): A dictionary containing a list of submission dictionaries under the key 'submissions'.
        Returns:
            pd.DataFrame: A DataFrame containing the processed submission data with the following columns:
            - id: The submission ID.
            - title: The title of the submission.
            - selftext: The text content of the submission.
            - author: The author of the submission.
            - created_utc: The creation time of the submission in UTC.
            - retrieved_utc: The retrieval time of the submission in UTC.
            - permalink: The permalink URL of the submission.
            - url: The URL of the submission.
            - score: The score of the submission.
            - over_18: Boolean indicating if the submission is marked as NSFW.
            - subreddit_id: The ID of the subreddit where the submission was posted.
            - subreddit: The name of the subreddit where the submission was posted.
            - comment_count: The number of comments on the submission.
        """
        # Assuming the response is a list of submission dictionaries
        submissions = response.get('submissions', [])
            
        processed_data = []
        for submission in submissions:
            processed_submission = {
                'id': submission.get('id'),
                'title': submission.get('title'),
                'selftext': submission.get('selftext'),
                'author': submission.get('author'),
                'created_utc': self._parse_unix_timestamp(submission.get('created_utc')),
                'retrieved_utc': self._parse_unix_timestamp(submission.get('retrieved_utc')),
                'permalink': submission.get('permalink'),
                'url': submission.get('url'),
                'score': int(submission.get('score', 0)),
                'over_18': submission.get('over_18', False),
                'subreddit_id': submission.get('subreddit_id'),
                'subreddit': submission.get('subreddit'),
                'comment_count': int(submission.get('comment_count', 0))
            }
            processed_data.append(processed_submission)
        
        return pd.DataFrame(processed_data)

    def search_submissions(self, query, author,start, end, score_min, score_max, subreddit, subreddit_id, comments_min, comments_max, over18, method = 'keyword',search_in = 'all', limit = 1000, ):
        """
        Search for submissions based on various criteria.

        Parameters:
        query (str): The search query string.
        author (str): The author of the submissions.
        method (str): The search method to use. Keyword or semantic.
        start (int): The start timestamp for the search range.
        end (int): The end timestamp for the search range.
        score_min (int): The minimum score of the submissions.
        score_max (int): The maximum score of the submissions.
        subreddit (str): The subreddit to search in.
        subreddit_id (str): The ID of the subreddit to search in.
        limit (int): The maximum number of submissions to return.
        search_in (str): The fields to search in.
        restricted (bool): search for NSFW or SFW submissions or both.
        comments_min (int): The minimum number of comments.
        comments_max (int): The maximum number of comments.

        Returns:
        pd.DataFrame: A DataFrame containing the search results.

        Raises:
        APIError: If there is an error with the API request.
        """
        params = {
            'query': query,
            'author': author,
            'method': method,
            'start': start,
            'end': end,
            'score_min': score_min,
            'score_max': score_max,
            'subreddit': subreddit,
            'subreddit_id': subreddit_id,
            'limit': limit,
            'search_in': search_in,
            'over18': over18,
            'comments_min': comments_min,
            'comments_max': comments_max
        }
        try:
            response = self._make_request('submissions', params=params)
            return self._process_submission_response(response)
        except APIError as e:
            print(f"Error searching submissions: {e}")
            return pd.DataFrame()
        
    def _process_comment_response(self, response):
    # Extract the comments from the response
        comments = response.get('comments', [])
    
    # Process each comment
        processed_data = []
        for comment in comments:
            processed_comment = {
            'id': comment.get('id'),
            'author': comment.get('author'),
            'body': comment.get('body'),
            'created_utc': datetime.fromtimestamp(comment.get('created_utc'), tz=timezone.utc),
            'submission_id': comment.get('submission_id'),
            'parent_id': comment.get('parent_id'),
            'score': int(comment.get('score', 0)),
            'subreddit_id': comment.get('subreddit_id'),
            'subreddit': comment.get('subreddit'),
            'permalink': comment.get('permalink'),
            'retrieved_on': datetime.fromtimestamp(comment.get('retrieved_on'), tz=timezone.utc) if comment.get('retrieved_on') else pd.NaT
        }
            processed_data.append(processed_comment)
    
    # Convert the list of dictionaries to a pandas DataFrame
        df = pd.DataFrame(processed_data)
    
    # Ensure all columns are present, even if no data is available
        expected_columns = ['id', 'author', 'body', 'created_utc', 'submission_id', 'parent_id', 
                        'score', 'subreddit_id', 'subreddit', 'permalink', 'retrieved_on']
        for col in expected_columns:
            if col not in df.columns:
                df[col] = pd.NA
    
    # Reorder columns to match the expected order
        df = df[expected_columns]
    
        return df
        
    def search_comments(self, query, author= None, start= None, end= None, score_min= None, score_max= None, subreddit= None, subreddit_id = None, over18 = None, method = 'keyword',search_in = 'all', limit = 1000, ):
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
        start : int
            The start timestamp for the search range.
        end : int
            The end timestamp for the search range.
        score_min : int
            The minimum score of the comments.
        score_max : int
            The maximum score of the comments.
        subreddit : str
            The subreddit to search in.
        subreddit_id : str
            The ID of the subreddit to search in.
        limit : int
            The maximum number of comments to return.
        search_in : str
            The fields to search in.
        restricted : bool
            Whether to restrict the search to certain criteria. NSFW or SFW comments or both.
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
            'start': start,
            'end': end,
            'score_min': score_min,
            'score_max': score_max,
            'subreddit': subreddit,
            'subreddit_id': subreddit_id,
            'limit': limit,
            'search_in': search_in,
            'over18': over18,
                
        }

        try:
            response = self._make_request('comments', params=params)
            return self._process_comment_response(response)
        except APIError as e:
            print(f"Error searching comments: {e}")
            return pd.DataFrame()
        
    def get_comments(self, submission_ids):
        try:
            response = self._make_request(f'comments', params ={'submission_ids': submission_ids})
            return self._process_comment_response(response)
        except APIError as e:
            print(f"Error getting comments: {e}")
            return pd.DataFrame()    