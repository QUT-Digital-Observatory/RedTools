#reddit topic trees

import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Any, List, Dict, Optional, Union
from datetime import datetime
import time
from bertopic import BERTopic
from typing import Tuple
import networkx as nx
import praw
import yaml
import re
import matplotlib.pyplot as plt
from LDA_over_time import LDA_over_time
from emo_intensity_over_time import EmoIntensityOverTime

from utils import (
    load_config,
    expand_dataframe_with_sentences,
    processed_text_column,
    remove_stopwords,
    create_umap_hdbscan_models,
)

class Reddit_trees:
    def __init__(self, config_path='config.yaml'):
        # Load the configuration
        self.config = load_config(config_path)
        self.hardware = self.config.get('hardware', 'CPU')

        # Initialize Reddit instance (optional — only needed for data collection)
        reddit_cfg = self.config.get('reddit', {})
        if reddit_cfg.get('client_id') and reddit_cfg.get('client_secret'):
            self.reddit = praw.Reddit(
                client_id=reddit_cfg['client_id'],
                client_secret=reddit_cfg['client_secret'],
                redirect_uri=reddit_cfg.get('redirect_uri', ''),
                user_agent=reddit_cfg.get('user_agent', 'RedTools')
            )
        else:
            self.reddit = None

        # Initialize LDA modeling
        self.lda_modeling = LDA_over_time()
    
    def search_subreddit(self, query, subreddit="australia", sort="new"):
        sub = self.reddit.subreddit(subreddit)
        result = sub.search(query, sort=sort)

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
                "time_created": datetime.fromtimestamp(submission.created_utc).isoformat(),
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
                    'time_created': datetime.fromtimestamp(comment.created_utc).isoformat(),
                    "link_id": comment.link_id,
                    "parent_id": comment.parent_id,
                    "replies": reply_ids,
                    "reply_count": len(reply_ids)
                }
                comments_data.append(comment_dict)

            # Sleep for a few seconds between each submission to avoid hitting rate limits
            time.sleep(2)

        return pd.DataFrame(comments_data)
    
    def format_url_for_search(self, url: Any) -> Union[str, None]:
        if url is None:
            return None
        try:
            # Convert to string if it's not already
            url_str = str(url).strip()
            if not url_str:
                return None
            # Remove 'https://' or 'http://' from the URL
            url_str = re.sub(r'^https?://', '', url_str)
            # Escape any special characters in the URL
            url_str = re.escape(url_str)
            return f"url:'{url_str}'"
        except Exception as e:
            print(f"Error formatting URL {url}: {str(e)}")
            return None

    def search_urls_for_ids(self, urls: List[Any], subreddit: str, sort: str) -> pd.DataFrame:
        all_submissions = []
        
        for url in tqdm(urls, desc="Searching URLs"):
            formatted_query = self.format_url_for_search(url)
            if not formatted_query:
                print(f"Skipping empty or invalid URL: {url}")
                continue
            
            try:
                # Search for the specific URL
                search_results = self.reddit.subreddit(subreddit).search(formatted_query, sort=sort, limit=None)
                
                # Collect submission information
                for submission in search_results:
                    submission_info = {
                        "id": submission.id,
                        "search_url": str(url)
                    }
                    all_submissions.append(submission_info)
                
            except Exception as e:
                print(f"Error searching for URL {url}: {str(e)}")
        
        # Convert the list of dictionaries to a DataFrame
        return pd.DataFrame(all_submissions)
    
    def save_to_file(self, df: pd.DataFrame, filename: str, file_format: str = 'csv'):
        if file_format == 'csv':
            df.to_csv(filename, index=False)
        elif file_format == 'excel':
            df.to_excel(filename, index=False)
        elif file_format == 'parquet':
            df.to_parquet(filename, engine='pyarrow')
        else:
            raise ValueError("Invalid file format. Choose from 'csv', 'excel' or 'parquet'.")
        
    def expand_dataframe_with_sentences(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Expands a DataFrame by splitting text into sentences and preprocessing them."""
        return expand_dataframe_with_sentences(df, text_column)

    def processed_text_column(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Preprocesses the text in the specified column of the input DataFrame."""
        return processed_text_column(df, text_column)

    def remove_stopwords(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Removes stopwords from the text in the specified column of the input DataFrame."""
        return remove_stopwords(df, text_column)

    def topic_model_comments(self, comments, text_column="body"):
        umap_model, hdbscan_model = create_umap_hdbscan_models(self.config, self.hardware)
        bertopic_params = self.config['bertopic']
        topic_model = BERTopic(
                    umap_model=umap_model,
                    hdbscan_model=hdbscan_model,
                    calculate_probabilities=bertopic_params['calculate_probabilities'],
                    verbose=bertopic_params['verbose'],
                    min_topic_size=bertopic_params['min_topic_size'])
        topics, _ = topic_model.fit_transform(comments[text_column])
        docs = topic_model.get_document_info(comments[text_column], df = comments)
        topic_list = topic_model.get_topic_info()
        return docs, topic_list
    
    def topic_model_submissions(self, submissions, text_column_1="title", text_column_2="selftext"):
        umap_model, hdbscan_model = create_umap_hdbscan_models(self.config, self.hardware)
        bertopic_params = self.config['bertopic']
        topic_model = BERTopic(
                    umap_model=umap_model,
                    hdbscan_model=hdbscan_model,
                    calculate_probabilities=bertopic_params['calculate_probabilities'],
                    verbose=bertopic_params['verbose'],
                    min_topic_size=bertopic_params['min_topic_size'])
        combined_text = submissions[text_column_1] + submissions[text_column_2]
        topics, _ = topic_model.fit_transform(combined_text)
        docs = topic_model.get_document_info(combined_text, df = submissions)
        topic_list = topic_model.get_topic_info()
        return docs, topic_list
    
    def topic_model_combined(self, comments: pd.DataFrame, submissions: pd.DataFrame, text_column: str = 'body', text_column_1: str = "title", text_column_2: str = "selftext"):
        # Ensure that the input columns are strings
        comments[text_column] = comments[text_column].astype(str)
        submissions[text_column_1] = submissions[text_column_1].astype(str)
        submissions[text_column_2] = submissions[text_column_2].astype(str)

        # Combine text from comments and submissions
        combined_text = comments[text_column].tolist() + (submissions[text_column_1] + submissions[text_column_2]).tolist()

        # Initialize the models
        umap_model, hdbscan_model = create_umap_hdbscan_models(self.config, self.hardware)
        bertopic_params = self.config['bertopic']
        topic_model = BERTopic(
                    umap_model=umap_model,
                    hdbscan_model=hdbscan_model,
                    calculate_probabilities=bertopic_params['calculate_probabilities'],
                    verbose=bertopic_params['verbose'],
                    min_topic_size=bertopic_params['min_topic_size'])

        # Fit the model
        topics, _ = topic_model.fit_transform(combined_text)

        # Get document information
        docs = topic_model.get_document_info(combined_text, df=pd.concat([comments, submissions]))

        # Get topic information
        topic_list = topic_model.get_topic_info()

        return docs, topic_list
    
    def lda_comments(self, df, text_column, num_topics):
        lda_model, doc_term_matrix = self.lda_modeling.create_lda_model(df[text_column], num_topics)
        topic_words = self.lda_modeling.get_topic_words()
        df['Topic'] = self.lda_modeling.assign_topics_to_docs(doc_term_matrix)
        df['topic_words'] = df['Topic'].map(topic_words)
        return df, lda_model
    
    def lda_submissions(self, submissions, text_column_1="title", text_column_2="selftext", num_topics=10):
        combined_text = submissions[text_column_1] + submissions[text_column_2]
        lda_model, doc_term_matrix = self.lda_modeling.create_lda_model(combined_text, num_topics)
        topic_words = self.lda_modeling.get_topic_words()
        submissions['Topic'] = self.lda_modeling.assign_topics_to_docs(doc_term_matrix)
        submissions['topic_words'] = submissions['Topic'].map(topic_words)
        return submissions, lda_model
    
    def lda_combined(self, comments: pd.DataFrame, submissions: pd.DataFrame, text_column: str = 'body', text_column_1: str = "title", text_column_2: str = "selftext", num_topics: int = 10) -> pd.DataFrame:
        # Ensure that the input columns are strings
        comments[text_column] = comments[text_column].astype(str)
        submissions[text_column_1] = submissions[text_column_1].astype(str)
        submissions[text_column_2] = submissions[text_column_2].astype(str)
    
        # Combine text from comments and submissions
        combined_text = comments[text_column].tolist() + (submissions[text_column_1] + submissions[text_column_2]).tolist()
    
        # Create the LDA model
        lda_model, doc_term_matrix = self.lda_modeling.create_lda_model(combined_text, num_topics)
    
        # Get the topic words
        topic_words = self.lda_modeling.get_topic_words()
    
        # Assign topics to documents
        all_assigned_topics = self.lda_modeling.assign_topics_to_docs(doc_term_matrix)
    
        # Split assigned topics for comments and submissions
        comments['Topic'] = all_assigned_topics[:len(comments)]
        submissions['Topic'] = all_assigned_topics[len(comments):]
    
        # Map topic words to the assigned topics
        comments['topic_words'] = comments['Topic'].map(topic_words)
        submissions['topic_words'] = submissions['Topic'].map(topic_words)
    
        # Add a column to indicate the source of the data
        comments['source'] = 'comment'
        submissions['source'] = 'submission'
    
        # Combine the comments and submissions DataFrames
        combined_df = pd.concat([comments, submissions], ignore_index=True)
    
        return combined_df
    
    def sentiment_intensity(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        emo_intensity = EmoIntensityOverTime()
        lexicon_path = self.config['lexicon']['lexicon_filepath']
        lexicon = emo_intensity.load_lexicon(lexicon_path)
        # Work on a copy of the text column so the original text is preserved
        processed_col = f'{text_column}_processed'
        df[processed_col] = df[text_column]
        self.processed_text_column(df, processed_col)
        df = emo_intensity.analyse_sentences(df, lexicon, processed_col)
        df = df.drop(columns=[processed_col])
        return df
    
        
    def tree_graph_and_adj_list(self, df: pd.DataFrame, incl_topic: bool = True, topic_column: str = 'Topic',
                               id_col: str = 'id', author_col: str = 'author',
                               body_col: str = 'body', link_id_col: str = 'link_id',
                               parent_id_col: str = 'parent_id',
                               time_col: str = 'created_utc', time_is_utc: bool = True,
                               extra_node_cols: Optional[List[str]] = None) -> Tuple[nx.DiGraph, pd.DataFrame]:
        """Build a directed graph and adjacency list from a comment/submission DataFrame.

        Args:
            df: DataFrame containing the data.
            incl_topic: Whether to include a topic attribute on each node.
            topic_column: Column name for topic assignment.
            id_col, author_col, body_col, link_id_col, parent_id_col, time_col:
                Column name mappings — override these for non-Reddit schemas
                (e.g. for the _nt schema pass id_col='commentId', author_col='username',
                body_col='text', link_id_col='threadId', parent_id_col='responseTo',
                time_col='date', time_is_utc=False).
            time_is_utc: If True, convert the time column from a Unix timestamp
                via datetime.fromtimestamp(). If False, use the value as-is.
            extra_node_cols: Optional list of additional column names to include
                as node attributes (e.g. emotion scores).
        """
        # Create directed graph
        G_tree = nx.DiGraph()

        # Add nodes to the graph
        for index, row in df.iterrows():
            node_id = row[id_col]
            time_created = (datetime.fromtimestamp(row[time_col]).isoformat()
                            if time_is_utc else row[time_col])
            node_data = {
                'author': row.get(author_col, 'unknown'),
                'body': row.get(body_col, ''),
                'link_id': row[link_id_col],
                'time_created': time_created
            }
            if incl_topic:
                node_data['topic'] = row[topic_column]
            if extra_node_cols:
                for col in extra_node_cols:
                    node_data[col] = row[col]

            G_tree.add_node(node_id, **node_data)

        # Add edges to the graph and build adjacency list data
        adj_data = {'Source': [], 'Target': [], 'TimeCreated': [], 'LinkID': []}
        for index, row in df.iterrows():
            parent_id = str(row[parent_id_col]) if pd.notna(row[parent_id_col]) else ''
            source = parent_id.replace('t3_', '').replace('t1_', '')
            targets = row[id_col]
            time_created = (datetime.fromtimestamp(row[time_col]).isoformat()
                            if time_is_utc else row[time_col])
            link_id = row[link_id_col]

            if source and source in G_tree.nodes and targets in G_tree.nodes:
                G_tree.add_edge(source, targets, time_created=time_created, link_id=link_id)
                adj_data['Source'].append(source)
                adj_data['Target'].append(targets)
                adj_data['TimeCreated'].append(time_created)
                adj_data['LinkID'].append(link_id)

        # Create adjacency list DataFrame
        adj_list_df_tree = pd.DataFrame(adj_data)

        return G_tree, adj_list_df_tree
    
       

    def plot_basic_graph(self, G: nx.DiGraph, title: str):
        # Plot the graph
        plt.figure(figsize=(20, 20))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_size=50, font_size=8, font_color='black')
        plt.title(title)
        plt.show()

    def plot_kk_graph(self, G: nx.DiGraph, title: str):
        # Plot the graph
        plt.figure(figsize=(20, 20))
        pos = nx.kamada_kawai_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=50, font_size=8, font_color='black')
        plt.title(title)
        plt.show()        

    def save_graph(self, G: nx.DiGraph, filename: str):
        nx.write_graphml(G, filename)

    def save_adj_list(self, df: pd.DataFrame, filename: str):
        df.to_csv(filename, index=False)

    def count_branches_and_ends(self, G: nx.DiGraph) -> Tuple[int, int]:
        # Initialize counters
        branches = 0
        ends = 0

        # Iterate through the nodes
        for node in G.nodes:
            successors = list(G.successors(node))
            if len(successors) > 1:
                branches += 1
            elif len(successors) == 0:
                ends += 1

        return branches, ends    