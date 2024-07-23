# Hierarchical topics

import numpy as np
from openai import models
import pandas as pd
import re
from bertopic import BERTopic
import yaml
from datetime import datetime
from sentence_splitter import SentenceSplitter
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import plotly.express as px
import plotly.graph_objects as go
from tqdm.auto import tqdm
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from topic_window import TopicWindow
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px


def load_config(config_path: str) -> dict:
    """
    Load the configuration file from the specified path.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config_path = '/home/fleetr/RedTools/config.yaml'

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

class HierarchicalTopics:

    def __init__(self, config_path: str):
        self.models = []
        self.topic_windows = TopicWindow(config_path)
        self.bert_topic = BERTopic()
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def get_hierarchical_topics(self, data, text_column, date_column, timescale):

        hierarchical_topics = []
        topic_trees = []
        figs = []
        data_exp = self.topic_windows.expand_dataframe_with_sentences(data, text_column)
        frames = self.topic_windows.get_frames(data_exp, date_column, timescale)
        for frame in tqdm(frames):
            model = self.bert_topic.fit(frame[text_column])
            docs: list = frame[text_column].tolist()
            h_tops =model.hierarchical_topics(docs)
            hierarchical_topics.append(h_tops)
            t_trees = model.get_topic_tree(h_tops)
            topic_trees.append(t_trees)
            fig = model.visualize_hierarchy()
            figs.append(fig)
        return hierarchical_topics, topic_trees, figs    

    def get_topic_embeddings(self, data, text_column, date_column, timescale):
        topic_embeddings = []
        interval_labels = []
        data_exp = self.topic_windows.expand_dataframe_with_sentences(data, text_column)
        frames = self.topic_windows.get_frames(data_exp, date_column, timescale)
        for i, frame in enumerate(tqdm(frames)):
            model = self.bert_topic.fit(frame[text_column])
            all_topics = sorted(list(model.get_topics().keys()))
            freq_df = model.get_topic_freq()
            freq_df = freq_df.loc[freq_df.Topic != -1, :]
            topics = freq_df.Topic.tolist()
            indices = np.array([all_topics.index(topic) for topic in topics])
            embeddings = model.topic_embeddings_[indices]
            topic_embeddings.append(np.array(embeddings))
            interval_labels.extend([f"{timescale.capitalize()} {i+1}"] * len(embeddings))
        return topic_embeddings, interval_labels
    
    def combine_topic_vectors(self, topic_embeddings_list):
        all_topic_vectors = np.vstack(topic_embeddings_list)
        return all_topic_vectors

    def reduce_dimensionality(self, topic_vectors, method='pca'):
        if method == 'pca':
            pca = PCA(n_components=2)
            reduced_vectors = pca.fit_transform(topic_vectors)
        elif method == 'tsne':
            tsne = TSNE(n_components=2, random_state=42)
            reduced_vectors = tsne.fit_transform(topic_vectors)
        else:
            raise ValueError("Invalid method. Use 'pca' or 'tsne'.")
        return reduced_vectors

    def calculate_similarity(self, topic_vectors):
        return cosine_similarity(topic_vectors)

    def visualize_topics(self, reduced_vectors, interval_labels, method='pca'):
        df = pd.DataFrame(reduced_vectors, columns=[f'{method.upper()}1', f'{method.upper()}2'])
        df['interval'] = interval_labels
        
        if method == 'pca':
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(df[f'{method.upper()}1'], df[f'{method.upper()}2'], c=pd.Categorical(df['interval']).codes, cmap='viridis')
            plt.colorbar(scatter, ticks=range(len(set(interval_labels))), label='Intervals')
            for i, interval in enumerate(df['interval']):
                plt.annotate(interval, (df[f'{method.upper()}1'][i], df[f'{method.upper()}2'][i]))
            plt.title(f'Topic Evolution Over Time ({method.upper()})')
            plt.xlabel(f'{method.upper()} Component 1')
            plt.ylabel(f'{method.upper()} Component 2')
            plt.show()
        elif method == 'tsne':
            fig = px.scatter(df, x=f'{method.upper()}1', y=f'{method.upper()}2', color='interval', text='interval',
                             title=f'Topic Evolution Over Time ({method.upper()})')
            fig.update_traces(textposition='top center')
            fig.update_layout(xaxis_title=f'{method.upper()} Component 1', yaxis_title=f'{method.upper()} Component 2')
            fig.show()
        else:
            raise ValueError("Invalid method. Use 'pca' or 'tsne'.")


