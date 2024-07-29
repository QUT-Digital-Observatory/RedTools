#Acquisitions AusReddit api wrapper - aaraw

import requests
import json
import time
import os
import pandas as pd
from datetime import datetime, timezone

base_url = 'https://ausreddit.digitialobservatory.net.au/api/v1'

class APIError(Exception):
    """Custom exception class to handle API errors"""
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        super().__init__(f"API Error {status_code}: {message}")

class APIWrapper:
    def __init__(self, base_url, jwt_file_path):
        self.base_url = base_url
        self.jwt_file_path = jwt_file_path
        self.session = requests.Session()
        self.token = None
        self.token_expiry = None

    def _load_jwt(self):
        if not os.path.exists(self.jwt_file_path):
            raise FileNotFoundError(f"JWT file not found at {self.jwt_file_path}")
        
        with open(self.jwt_file_path, 'r') as file:
            jwt_data = json.load(file)
        
        self.token = jwt_data.get('token')
        expiry_str = jwt_data.get('expiry')
        if expiry_str:
            self.token_expiry = datetime.fromisoformat(expiry_str)
        
        if not self.token:
            raise ValueError("No token found in JWT file")

    def _is_token_valid(self):
        if not self.token or not self.token_expiry:
            return False
        return datetime.now() < self.token_expiry

    def _get_auth_header(self):
        if not self._is_token_valid():
            self._load_jwt()
        return {"Authorization": f"Bearer {self.token}"}
    
    def _make_request(self, endpoint, params=None, method='GET', data=None):
        url = f"{self.base_url}/{endpoint}"
        headers = self._get_auth_header()
        
        try:
            if method == 'GET':
                response = self.session.get(url, params=params, headers=headers)
            elif method == 'POST':
                response = self.session.post(url, params=params, headers=headers, json=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            # Here, we're sure that 'response' exists because HTTPError is only raised after a response is received
            if response and response.status_code == 422:  # type: requests.Response
                error_detail = response.json().get('detail', 'No detail provided')
                raise APIError(422, f"Unprocessable Entity: {error_detail}")
            elif response and 400 <= response.status_code < 500:
                raise APIError(response.status_code, f"Client Error: {response.text}")
            elif response and 500 <= response.status_code < 600:
                raise APIError(response.status_code, f"Server Error: {response.text}")
            else:
                raise APIError(http_err.response.status_code, f"HTTP Error: {str(http_err)}")
        except requests.exceptions.ConnectionError as conn_err:
            raise APIError(None, f"Connection Error: {str(conn_err)}")
        except requests.exceptions.Timeout as timeout_err:
            raise APIError(None, f"Timeout Error: {str(timeout_err)}")
        except requests.exceptions.RequestException as req_err:
            raise APIError(None, f"Request Error: {str(req_err)}")

    def list_subreddits(self, meta=False):
        try:
            response = self._make_request('subreddits', params={'meta': meta})
            return self._process_subreddit_response(response)
        except APIError as e:
            print(f"Error listing subreddits: {e}")
            return pd.DataFrame()  # Return an empty DataFrame on error

    def _process_subreddit_response(self, response):
        # Assuming the response is a list of subreddit dictionaries
        subreddits = response.get('subreddits', [])  # Adjust this based on your actual API response structure
        
        processed_data = []
        for subreddit in subreddits:
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
        if timestamp is not None:
            return datetime.fromtimestamp(int(timestamp), tz=timezone.utc)
        return None

    def get_submissions(self, subreddit_id: str):
        try:
            response = self._make_request(f'submissions/{subreddit_id}')
            return self._process_submission_response(response)
        except APIError as e:
            print(f"Error getting submissions: {e}")
            return pd.DataFrame()
        
    def _process_submission_response(self, response):
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
    
    def get_comments(self, submission_id):
        try:
            response = self._make_request(f'comments/{submission_id}')
            return self._process_comment_response(response)
        except APIError as e:
            print(f"Error getting comments: {e}")
            return pd.DataFrame()
        
    def _process_comment_response(self, response):
        # Assuming the response is a list of comment dictionaries
        comments = response.get('comments', [])
            
        processed_data = []
        for comment in comments:
            processed_comment = {
                'id': comment.get('id'),
                'author': comment.get('author'),
                'body': comment.get('body'),
                'created_utc': self._parse_unix_timestamp(comment.get('created_utc')),
                'link_id': comment.get('link_id'),
                'parent_id': comment.get('parent_id'),
                'score': int(comment.get('score', 0)),
                'subreddit_id': comment.get('subreddit_id'),
                'subreddit': comment.get('subreddit'),
                'permalink': comment.get('permalink'),
                'retrieved_utc': self._parse_unix_timestamp(comment.get('retrieved_utc'))
            }
            processed_data.append(processed_comment)
        
        return pd.DataFrame(processed_data)
