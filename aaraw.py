#Acquisitions AusReddit api wrapper - aaraw

import requests
import json
import time
import os
import pandas as pd
from datetime import datetime, timedelta

base_url = 'https://ausreddit.digitialobservatory.net.au/api/v1'

class AusReddit:

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
        
        if method == 'GET':
            response = self.session.get(url, params=params, headers=headers)
        elif method == 'POST':
            response = self.session.post(url, params=params, headers=headers, json=data)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        response.raise_for_status()
        return response.json()
    
    def list_subreddits(self, meta=False):
        response = self._make_request('subreddits', params={'meta': meta})
        return self._process_subreddit_response(response)

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
            return datetime.utcfromtimestamp(int(timestamp))
        return None
