#Acquisitions AusReddit api wrapper - aaraw

import requests
import json
import os
import pandas as pd
from datetime import datetime, timezone

from utils import (
    APIError,
    parse_unix_timestamp,
    process_subreddit_response,
    process_submission_response,
    process_comment_response,
    make_api_request,
)

base_url = 'https://ausreddit.digitialobservatory.net.au/api/v1'

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
        # Ensure token is valid before making the request
        self._get_auth_header()
        return make_api_request(self.session, self.base_url, endpoint, self.token, params, method, data)

    def list_subreddits(self, meta=False):
        try:
            response = self._make_request('subreddits', params={'meta': meta})
            return self._process_subreddit_response(response)
        except APIError as e:
            print(f"Error listing subreddits: {e}")
            return pd.DataFrame()  # Return an empty DataFrame on error

    def _process_subreddit_response(self, response):
        return process_subreddit_response(response)

    def _parse_unix_timestamp(self, timestamp):
        return parse_unix_timestamp(timestamp)

    def get_submissions(self, subreddit_id: str):
        try:
            response = self._make_request(f'submissions/{subreddit_id}')
            return self._process_submission_response(response)
        except APIError as e:
            print(f"Error getting submissions: {e}")
            return pd.DataFrame()
        
    def _process_submission_response(self, response):
        return process_submission_response(response)
    
    def get_comments(self, submission_id):
        try:
            response = self._make_request(f'comments/{submission_id}')
            return self._process_comment_response(response)
        except APIError as e:
            print(f"Error getting comments: {e}")
            return pd.DataFrame()
        
    def _process_comment_response(self, response):
        return process_comment_response(response)

    def search_submissions(self, query, author, method, start, end, score_min, score_max, subreddit, subreddit_id, limit, context, search_in, restricted, comments_min, comments_max):
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
            'context': context,
            'search_in': search_in,
            'restricted': restricted,
            'comments_min': comments_min,
            'comments_max': comments_max
        }
        try:
            response = self._make_request('search/submissions', params=params)
            return self._process_submission_response(response)
        except APIError as e:
            print(f"Error searching submissions: {e}")
            return pd.DataFrame()
        
    def search_comments(self, query, author, method, start, end, score_min, score_max, subreddit, subreddit_id, limit, context, search_in, restricted):
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
            'context': context,
            'search_in': search_in,
            'restricted': restricted,
                
        }
        try:
            response = self._make_request('search/comments', params=params)
            return self._process_comment_response(response)
        except APIError as e:
            print(f"Error searching comments: {e}")
            return pd.DataFrame()