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
from RedTools import topic_window
from topic_window import TopicWindow


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

class HierarchicalTopics:

    def __init__(self, config_path: str):
        self.models = []
        self.topic_windows = TopicWindow(config_path)
        self.bert_topic = BERTopic()
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def get_hierarchical_topics(self, data, text_column, date_column, timescale, window_size):

        hierarchical_topics = []
        data_exp = self.topic_windows.expand_dataframe_with_sentences(data, text_column)
        frames = self.topic_windows.get_frames(data_exp, date_column, timescale)
        for frame in frames:
            model = self.bert_topic.fit(frame[text_column])
            docs: list = frame[text_column].tolist()
            hierarchical_topics =model.hierarchical_topics(docs)

