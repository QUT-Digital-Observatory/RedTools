#reddit topic trees

import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Dict
from datetime import datetime
import time
from bertopic import BERTopic
from typing import Tuple
import networkx as nx
import praw
import yaml
from sentence_splitter import SentenceSplitter
import re
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from LDA_over_time import LDA_over_time
from emo_intensity_over_time import EmoIntensityOverTime

def load_config(config_path: str) -> dict:
    """
    Load the configuration file from the specified path.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config_path = 'config.yaml'

config = load_config(config_path)
hardware = config.get('hardware', 'GPU')

if hardware == 'GPU':
    from cuml.manifold import UMAP
    from cuml.cluster import HDBSCAN
    print("Using GPU for UMAP and HDBSCAN.")
else:
    from umap import UMAP
    from hdbscan import HDBSCAN
    print("Using CPU for UMAP and HDBSCAN.")


class Reddit_trees:
    def __init__(self, config_path='config.yaml'):
        # Load the configuration
        with open(config_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)
        
        # Initialize Reddit instance
        self.reddit = praw.Reddit(
            client_id=self.config['reddit']['client_id'],
            client_secret=self.config['reddit']['client_secret'],
            redirect_uri=self.config['reddit']['redirect_uri'],
            user_agent=self.config['reddit']['user_agent']
        )
        
        # Initialize LDA modeling
        self.lda_modeling = LDA_over_time()
    
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
        
    def __pre_process__(self, text: str) -> str:  
        text = re.sub(r"&gt;", "", text)
        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        text = text.replace("\n", "").replace("\t", "").strip()
        return text
    
    def __sentence_chunker__(self, text: str) -> list:
    
        splitter = SentenceSplitter(language="en")
        return splitter.split(text)
    
    #You can process 1) a dataframe that you want to split into sentences and pre_process, 2) a dataframe that is not split into sentences but is pre_processed, 3) a dataframe that is not split into sentences and is not pre_processed

    def expand_dataframe_with_sentences(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Expands a DataFrame by splitting the text in a specified column into sentences,
        preprocesses them, and each sentence retains metadata from the original row.
        """
        # Apply sentence_chunker to the text column and explode the result into new rows
        df[text_column] = df[text_column].astype(str)
        df.dropna(subset=[text_column], inplace=True)
        df['sentences'] = df[text_column].apply(self.__sentence_chunker__)
        df_expanded = df.explode('sentences')

        df_expanded = df_expanded[df_expanded['sentences'].str.strip() != '']
        df_expanded['sentences'] = df_expanded['sentences'].apply(self.__pre_process__)
        df_expanded[text_column] = df_expanded['sentences']
        df_expanded.drop(columns=['sentences'], inplace=True)

        df_expanded = df_expanded[df_expanded[text_column].str.strip() != '' ]
        
        df_expanded.reset_index(drop=True, inplace=True)

        return df_expanded
    
    def processed_text_column(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Preprocesses the text in the specified column of the input DataFrame.
        """
        # Apply the pre_process function to the text column
        df[text_column] = df[text_column].astype(str)
        df[text_column] = df[text_column].apply(self.__pre_process__)
        return df

    def remove_stopwords(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Removes stopwords from the text in the specified column of the input DataFrame.
        """
        # Remove stopwords from the text column
        df[text_column] = df[text_column].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords.words('english'))]))
        return df    

    def topic_model_comments(self, comments, text_column="body"):
        umap_params = self.config['umap']
        hdbscan_params = self.config['hdbscan']
        bertopic_params = self.config['bertopic']
        # Initialize models with parameters from config
        if hardware == 'GPU':
            umap_model = UMAP(
            n_components=umap_params['n_components'],
            n_neighbors=umap_params['n_neighbors'],
            min_dist=umap_params['min_dist'],
            random_state=umap_params['random_state']
        )
            hdbscan_model = HDBSCAN(
            min_samples=hdbscan_params['min_samples'],
            gen_min_span_tree=hdbscan_params['gen_min_span_tree'],
            prediction_data=hdbscan_params['prediction_data']
        )
        else:
            umap_model = UMAP()
            hdbscan_model = HDBSCAN()  

        umap_model = umap_model
        hdbscan_model = hdbscan_model
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
    
    def topic_model_submissions(self, submissions, text_column_1="selftext", text_column_2="title"):
        umap_params = self.config['umap']
        hdbscan_params = self.config['hdbscan']
        bertopic_params = self.config['bertopic']
        # Initialize models with parameters from config
        if hardware == 'GPU':
            umap_model = UMAP(
            n_components=umap_params['n_components'],
            n_neighbors=umap_params['n_neighbors'],
            min_dist=umap_params['min_dist'],
            random_state=umap_params['random_state']
        )
            hdbscan_model = HDBSCAN(
            min_samples=hdbscan_params['min_samples'],
            gen_min_span_tree=hdbscan_params['gen_min_span_tree'],
            prediction_data=hdbscan_params['prediction_data']
        )
        else:
            umap_model = UMAP()
            hdbscan_model = HDBSCAN() 
        umap_model = umap_model
        hdbscan_model = hdbscan_model
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
    
    def topic_model_combined(self, comments: pd.DataFrame, submissions: pd.DataFrame, text_column: str = 'body', text_column_1: str = "selftext", text_column_2: str = "title"):
        # Ensure that the input columns are strings
        comments[text_column] = comments[text_column].astype(str)
        submissions[text_column_1] = submissions[text_column_1].astype(str)
        submissions[text_column_2] = submissions[text_column_2].astype(str)

        # Combine text from comments and submissions
        combined_text = comments[text_column].tolist() + (submissions[text_column_1] + submissions[text_column_2]).tolist()

        # Initialize the models
        umap_params = self.config['umap']
        hdbscan_params = self.config['hdbscan']
        bertopic_params = self.config['bertopic']
        # Initialize models with parameters from config
        if hardware == 'GPU':
            umap_model = UMAP(
            n_components=umap_params['n_components'],
            n_neighbors=umap_params['n_neighbors'],
            min_dist=umap_params['min_dist'],
            random_state=umap_params['random_state']
        )
            hdbscan_model = HDBSCAN(
            min_samples=hdbscan_params['min_samples'],
            gen_min_span_tree=hdbscan_params['gen_min_span_tree'],
            prediction_data=hdbscan_params['prediction_data']
        )
        else:
            umap_model = UMAP()
            hdbscan_model = HDBSCAN()
            
        umap_model = umap_model
        hdbscan_model = hdbscan_model
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
        df['assigned_topic'] = self.lda_modeling.assign_topics_to_docs(doc_term_matrix)
        df['topic_words'] = df['assigned_topic'].map(topic_words)
        return df, lda_model
    
    def lda_submissions(self, submissions, text_column_1="selftext", text_column_2="title", num_topics=10):
        combined_text = submissions[text_column_1] + submissions[text_column_2]
        lda_model, doc_term_matrix = self.lda_modeling.create_lda_model(combined_text, num_topics)
        topic_words = self.lda_modeling.get_topic_words()
        submissions['assigned_topic'] = self.lda_modeling.assign_topics_to_docs(doc_term_matrix)
        submissions['topic_words'] = submissions['assigned_topic'].map(topic_words)
        return submissions, lda_model
    
    def lda_combined(self, comments: pd.DataFrame, submissions: pd.DataFrame, text_column: str = 'body', text_column_1: str = "selftext", text_column_2: str = "title", num_topics: int = 10) -> pd.DataFrame:
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
        comments['assigned_topic'] = all_assigned_topics[:len(comments)]
        submissions['assigned_topic'] = all_assigned_topics[len(comments):]
    
        # Map topic words to the assigned topics
        comments['topic_words'] = comments['assigned_topic'].map(topic_words)
        submissions['topic_words'] = submissions['assigned_topic'].map(topic_words)
    
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
        self.processed_text_column(df, text_column)
        df = emo_intensity.analyse_sentences(df, lexicon, text_column)
        return df
    
    def tree_graph_and_adj_list(self, df: pd.DataFrame, incl_topic: bool = True, topic_column = 'Topic') -> Tuple[nx.DiGraph, pd.DataFrame]:
        # Create directed graph
        G_tree = nx.DiGraph()

        # Add nodes to the graph
        for index, row in df.iterrows():
            node_id = row['id']
            author = row['author']
            body = row['body']
            link_id = row['link_id']
            time_created = datetime.fromtimestamp(row['created_utc']).isoformat()
            node_data = {
                'author': author,
                'body': body,
                'link_id': link_id,
                'time_created': time_created
            }
            if incl_topic:
                node_data['topic'] = row[topic_column]
            
            G_tree.add_node(node_id, **node_data)

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
    
    def tree_graph_and_adj_list_nt(self, df: pd.DataFrame, incl_topic: bool = True, topic_column = 'Topic') -> Tuple[nx.DiGraph, pd.DataFrame]:
        # Create directed graph
        G_tree = nx.DiGraph()

        # Add nodes to the graph
        for index, row in df.iterrows():
            node_id = row['commentId']
            author = row['username']
            body = row['text']
            link_id = row['threadId']
            time_created = row['date']
            node_data = {
                'author': author,
                'body': body,
                'link_id': link_id,
                'time_created': time_created
            }
            if incl_topic:
                node_data['topic'] = row[topic_column]
            
            G_tree.add_node(node_id, **node_data)

        # Add edges to the graph and build adjacency list data
        adj_data = {'Source': [], 'Target': [], 'TimeCreated': [], 'LinkID': []}
        for index, row in df.iterrows():
            parent_id = str(row['responseTo']) if pd.notna(row['responseTo']) else ''
            source = parent_id.replace('t3_', '').replace('t1_', '')
            targets = row['commentId']
            time_created = row['date']
            link_id = row['threadId']

            if source and targets in G_tree.nodes:
                G_tree.add_edge(source, targets, time_created=time_created, link_id=link_id)
                adj_data['Source'].append(source)
                adj_data['Target'].append(targets)
                adj_data['TimeCreated'].append(time_created)
                adj_data['LinkID'].append(link_id)

        # Create adjacency list DataFrame
        adj_list_df_tree = pd.DataFrame(adj_data)
    
        return G_tree, adj_list_df_tree

    def tree_graph_and_adj_list_emo(self, df) -> Tuple[nx.DiGraph, pd.DataFrame]:
        # Create directed graph
        G_tree = nx.DiGraph()

        # Add nodes to the graph
        for index, row in df.iterrows():
            node_id = row['id']
            author = row['username']
            body = row['text']
            link_id = row['threadId']
            time_created = row['date']
            node_data = {
                'author': author,
                'body': body,
                'link_id': link_id,
                'time_created': time_created,
                'anger': row['anger'],
                'anticipation': row['anticipation'],
                'disgust': row['disgust'],
                'fear': row['fear'],
                'joy': row['joy'],
                'sadness': row['sadness'],
                'surprise': row['surprise'],
                'trust': row['trust']
            }
            
            G_tree.add_node(node_id, **node_data)

        # Add edges to the graph and build adjacency list data
        adj_data = {'Source': [], 'Target': [], 'TimeCreated': [], 'LinkID': []}
        for index, row in df.iterrows():
            parent_id = str(row['responseTo']) if pd.notna(row['responseTo']) else ''
            source = parent_id.replace('t3_', '').replace('t1_', '')
            targets = row['commentId']
            time_created = row['date']
            link_id = row['threadId']

            if source and targets in G_tree.nodes:
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