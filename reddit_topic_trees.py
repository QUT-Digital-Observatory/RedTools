#reddit topic trees

 
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Dict
from datetime import datetime
import time
from bertopic import BERTopic
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP
from typing import Tuple
import networkx as nx
import praw

# Create a Reddit instance
reddit = praw.Reddit(client_id='voiLhjY_Q0uJwzVwU9Xbhg', 
                    client_secret='dP7Ws55HLQUoSYpXkOzkzEPjqyuGtg', 
                    redirect_uri='http://127.0.0.1:3000/profile',                     
                    user_agent='r_aus_astro by Due_recordings9516')


class Reddit_trees:
    def __init__(self):
        self.reddit = reddit 
    
    def search_subreddit(self, query, subreddit="australia", sort="new"):
        subreddit = self.reddit.subreddit(subreddit)
        result = subreddit.search(query, sort=sort)

        # Initialize an empty list to store dictionaries, where each dictionary represents a submission
        submissions_data: list[dict] = []

        # Iterate through each submission and add title and ID to the list as dictionaries
        for submission in tqdm(result):

            submission_dict: dict = {
                "title": submission.title,
                "id": submission.id,
                "comment_count": submission.num_comments,
                "selftext": submission.selftext,
                "created_utc": submission.created_utc,
                "created_time": datetime.fromtimestamp(submission.created_utc).isoformat(),
                "url": submission.url  
            }

            submissions_data.append(submission_dict)

        # Create and return a DataFrame from the list of dictionaries
        return pd.DataFrame(submissions_data)

    def get_reply_ids(self, commentforest) -> List[str]:
        # Get the list of comments from the comment forest
        comments = commentforest.list()
        # Collect and return the IDs of these comments
        return [comment.id for comment in comments]

    def fetch_comments(self, ids_list: List[str]) -> pd.DataFrame:
        comments_data: List[Dict[str, object]] = []

        for id_ in tqdm(ids_list):

            submission = self.reddit.submission(id=id_)
            submission.comments.replace_more(limit=None)

            for comment in tqdm(submission.comments.list()):
                reply_ids = self.get_reply_ids(comment.replies) if comment.replies else []
                comment_dict: Dict[str, object] = {
                    "author": str(comment.author),
                    "body": comment.body,
                    "id": comment.id,
                    "created_utc": comment.created_utc,
                    'tme_created': datetime.fromtimestamp(comment.created_utc).isoformat(),
                    "link_id": comment.link_id,
                    "parent_id": comment.parent_id,
                    "replies": reply_ids,
                    "reply_count": len(reply_ids)
            }
            
                comments_data.append(comment_dict)
        
        # Sleep for a few seconds between each request to avoid hitting rate limits
            time.sleep(2)

        return pd.DataFrame(comments_data)
    
    def save_to_file(self, df: pd.DataFrame, filename: str, file_format: str = 'csv'):
        if file_format == 'csv':
            df.to_csv(filename, index=False)
        elif file_format == 'excel':
            df.to_excel(filename, index=False)
        elif file_format == 'parquet':
            df.to_parquet(filename, engine='pyarrow')
        else:
            raise ValueError("Invalid file format. Choose from 'csv', 'excel' or 'parquet'.")

    def topic_model_comments(self, comments, text_column="body"):
        umap_model = UMAP(n_components=5, n_neighbors=10, min_dist=0.0, random_state=42)
        hdbscan_model = HDBSCAN(min_samples=5, gen_min_span_tree=True, prediction_data=True)
        topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, verbose=True)
        topics, _ = topic_model.fit_transform(comments[text_column])
        docs = topic_model.get_document_info(comments[text_column], df = comments)
        topic_list = topic_model.get_topic_info()
        return docs, topic_list
    
    def topic_model_submissions(self, submissions, text_column_1="selftext", text_column_2="title"):
        umap_model = UMAP(n_components=5, n_neighbors=10, min_dist=0.0, random_state=42)
        hdbscan_model = HDBSCAN(min_samples=5, gen_min_span_tree=True, prediction_data=True)
        topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, verbose=True)
        combined_text = submissions[text_column_1] + submissions[text_column_2]
        topics, _ = topic_model.fit_transform(combined_text)
        docs = topic_model.get_document_info(combined_text, df = submissions)
        topic_list = topic_model.get_topic_info()
        return docs, topic_list
    
    def topic_model_combined(self, comments: pd.DataFrame, submissions: pd.DataFrame, text_column: str = 'body', text_column_1: str = "selftext", text_column_2: str = "title"):
        # Ensure that the input columns are strings
        comments[text_column] = comments[text_column].astype(str)
        submissions[text_column_1] = submissions[text_column_1].astype(str)
        submissions[text_column_2] = submissions[text_column_2].astype(str)

        # Combine text from comments and submissions
        combined_text = comments[text_column].tolist() + (submissions[text_column_1] + submissions[text_column_2]).tolist()

        # Initialize the models
        umap_model = UMAP(n_components=5, n_neighbors=10, min_dist=0.0, random_state=42)
        hdbscan_model = HDBSCAN(min_samples=5, gen_min_span_tree=True, prediction_data=True)
        topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, verbose=True)

        # Fit the model
        topics, _ = topic_model.fit_transform(combined_text)

        # Get document information
        docs = topic_model.get_document_info(combined_text, df=pd.concat([comments, submissions]))

        # Get topic information
        topic_list = topic_model.get_topic_info()

        return docs, topic_list
    
    def tree_graph_and_adj_list(self, df: pd.DataFrame) -> Tuple[nx.DiGraph, pd.DataFrame]:
        # Create directed graph
        G_tree = nx.DiGraph()

        # Add nodes to the graph
        for index, row in df.iterrows():
            node_id = row['id']
            author = row['author']
            body = row['body']
            topic = row['Topic']
            link_id = row['link_id']
            time_created = datetime.fromtimestamp(row['created_utc']).isoformat()
            G_tree.add_node(node_id, author=author, body=body, topic=topic, link_id=link_id, time_created=time_created)

        # Add edges to the graph and build adjacency list data
        adj_data = {'Source': [], 'Target': [], 'TimeCreated': [], 'LinkID': []}
        for index, row in df.iterrows():
            parent_id = str(row['parent_id']) if pd.notna(row['parent_id']) else ''
            source = parent_id.replace('t3_', '').replace('t1_', '')
            targets = row['id']
            time_created = datetime.fromtimestamp(row['created_utc']).isoformat()
            link_id = row['link_id']

            if source and targets in G_tree.nodes:
                G_tree.add_edge(source, targets, time_created=time_created, link_id=link_id)
                adj_data['Source'].append(source)
                adj_data['Target'].append(targets)
                adj_data['TimeCreated'].append(time_created)
                adj_data['LinkID'].append(link_id)

            # Create adjacency list DataFrame
        adj_list_df_tree = pd.DataFrame(adj_data)
    
        return G_tree, adj_list_df_tree 
    

reddit_trees = Reddit_trees()

comments = reddit_trees.fetch_comments(['1alsvt7'])