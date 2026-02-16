#LDA over time

from typing import Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import AgglomerativeClustering
import scipy.sparse as sp

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from utils import (
    expand_dataframe_with_sentences,
    processed_text_column,
    remove_stopwords,
    get_frames,
)

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


#TODO: LDA hierarchical clustering working here. Plus vis trees over time
#TODO: Seeded LDA over time

class LDA_over_time:
    def __init__(self):
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def expand_dataframe_with_sentences(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Expands a DataFrame by splitting text into sentences and preprocessing them."""
        return expand_dataframe_with_sentences(df, text_column)

    def processed_text_column(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Preprocesses the text in the specified column of the input DataFrame."""
        return processed_text_column(df, text_column)

    def remove_stopwords(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Removes stopwords from the text in the specified column of the input DataFrame."""
        return remove_stopwords(df, text_column)

    def create_lda_model(self, texts, num_topics=5):
        self.vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        doc_term_matrix = self.vectorizer.fit_transform(texts)
        
        self.lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        self.lda_model.fit(doc_term_matrix)
        
        return self.lda_model, doc_term_matrix
    
    def get_topic_words(self, num_words=10):
        feature_names = self.vectorizer.get_feature_names_out()
        topic_words = {}
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_words_idx = topic.argsort()[:-num_words - 1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topic_words[topic_idx] = ', '.join(top_words)
        return topic_words

    def assign_topics_to_docs(self, doc_term_matrix):
        topic_assignments = self.lda_model.transform(doc_term_matrix)
        return topic_assignments.argmax(axis=1)

    def lda_with_dataframe(self, df, preprocessed_text_column, num_topics=5):
        lda_model, doc_term_matrix = self.create_lda_model(df[preprocessed_text_column], num_topics)
        topic_words = self.get_topic_words()
        df['assigned_topic'] = self.assign_topics_to_docs(doc_term_matrix)
        df['topic_words'] = df['assigned_topic'].map(topic_words)
        return df, lda_model
    
    def get_frames(self, data: pd.DataFrame, date_column: str, timescale: str = 'week') -> list:
        """Splits the DataFrame into intervals based on the specified timescale."""
        frames = get_frames(data, date_column, timescale)
        self.data = frames
        return frames
    
    def fit_model_to_frames(self, frames: list, text_column: str, num_topics: int = 5):
        """
        Fits an LDA model to each DataFrame slice in the input list of frames.
        """
        lda_models = []
        for frame in tqdm(frames, desc="Fitting LDA to frames"):
            lda_model, _ = self.create_lda_model(frame[text_column], num_topics)
            lda_models.append(lda_model)
        return lda_models
    
    def merge_frames_into_windows(self, frames, window_size, text_column: str = 'body'):
        """
        Merge frames into windows of specified size.
        """
        windows = []
        docs = []
        total_frames = len(frames)
        for i in tqdm(range(0, total_frames - window_size + 1), desc="Merging frames into windows"):
            window = pd.concat(frames[i:i + window_size]).reset_index(drop=True)
            docs = window[text_column].tolist()
            windows.append(window)
        return windows, docs
    
    def lda_model_for_windows(self, windows: list, text_column: str, num_topics: int = 5):
        """
        Fits an LDA model to each window in the input list of windows.
        """
        lda_results = []
        lda_models = []
        for window in tqdm(windows, desc="Fitting LDA to windows"):
            df, lda_model = self.lda_with_dataframe(window, text_column, num_topics)
            lda_results.append(df)
            lda_models.append(lda_model)
        return lda_results, lda_models

    def visualize_topic_trends(self, df, date_column, timescale='week'):
        """
        Visualize the trends of topics over time using a line plot.
        """
        # Ensure the date column is in datetime format
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Group by date and topic, count occurrences
        topic_counts = df.groupby([df[date_column].dt.to_period(timescale[0]), 'assigned_topic']).size().unstack(fill_value=0)
        
        # Create a line plot
        fig = px.line(topic_counts, x=topic_counts.index.astype(str), y=topic_counts.columns,
                      labels={'x': 'Time', 'y': 'Topic Frequency'},
                      title=f'Topic Trends over Time ({timescale.capitalize()})')
        
        return fig

    def visualize_topic_distribution(self, df):
        """
        Visualize the overall distribution of topics using a pie chart.
        """
        topic_dist = df['assigned_topic'].value_counts()
        fig = px.pie(values=topic_dist.values, names=topic_dist.index,
                     title='Overall Topic Distribution')
        return fig

    def visualize_word_cloud(self, topic_id):
        from wordcloud import WordCloud

        feature_names = self.vectorizer.get_feature_names_out()
        topic = self.lda_model.components_[topic_id]
        word_freq = dict(zip(feature_names, topic))

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for Topic {topic_id}')
        plt.show()

    #hierarchical LDAs
    
    def _preprocess_hlda(self, doc):
        tokens = word_tokenize(doc.lower())
        tokens = [word for word in tokens if word.isalnum()]
        tokens = [word for word in tokens if word not in self.stopwords]  
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return tokens
    
    def _get_top_words(self, model, feature_names, n_top_words=10):
        top_words = {}
        for topic_idx, topic in enumerate(model.components_):
            top_words[topic_idx] = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        return top_words
    
    def hlda_model(self, df: pd.DataFrame, text_column: str, n_topics: int = 10) -> pd.DataFrame:
        # Apply preprocessing to each document in the DataFrame
        processed_docs = df[text_column].apply(self._preprocess_hlda)
        processed_docs = processed_docs.apply(lambda x: ' '.join(x))
        
        # Vectorize the documents
        vectorizer = CountVectorizer()
        doc_term_matrix = vectorizer.fit_transform(processed_docs)
        
        # Perform LDA
        lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda_output = lda_model.fit_transform(doc_term_matrix)
        
        # Assign the most probable topic to each document
        most_probable_topics = lda_output.argmax(axis=1)
        
        # Get the top words for each topic
        feature_names = vectorizer.get_feature_names_out()
        top_words = self._get_top_words(lda_model, feature_names)
        
        # Map the most probable topic's words to each document
        topic_words = [', '.join(top_words[topic]) for topic in most_probable_topics]
        
        # Add the assigned topic and topic words to the DataFrame
        df['assigned_topic'] = most_probable_topics
        df['topic_words'] = topic_words
        
        # Hierarchical clustering of the topics
        topic_distributions = lda_model.components_
        clustering_model = AgglomerativeClustering(n_clusters=3)
        hierarchical_topics = clustering_model.fit_predict(topic_distributions)
        
        # Add hierarchical cluster information
        topic_hierarchy = {i: hierarchical_topics[i] for i in range(len(hierarchical_topics))}
        df['hierarchical_topic'] = df['assigned_topic'].map(topic_hierarchy)
        
        return df

    def _modify_term_matrix(self, term_matrix, vocab, seed_words, weight=10):
        for topic, words in seed_words.items():
            for word in words:
                if word in vocab:
                    word_index = vocab.tolist().index(word)
                    term_matrix[:, word_index] *= weight
        return term_matrix

    def seeded_lda_model(self, df: pd.DataFrame, text_column: str, seed_words: dict, n_topics: int = 10, n_iter: int = 1000) -> pd.DataFrame:
        # Apply preprocessing to each document in the DataFrame
        processed_docs = df[text_column].apply(self._preprocess_hlda)
        processed_docs = processed_docs.apply(lambda x: ' '.join(x))
        
        # Vectorize the documents
        vectorizer = CountVectorizer()
        doc_term_matrix = vectorizer.fit_transform(processed_docs)
        vocab = vectorizer.get_feature_names_out()
        
        # Modify the term matrix with seed words
        modified_term_matrix = self._modify_term_matrix(doc_term_matrix, vocab, seed_words)
        
        # Perform LDA
        lda_model = LatentDirichletAllocation(n_components=n_topics, max_iter=n_iter, random_state=42)
        lda_output = lda_model.fit_transform(modified_term_matrix)
        
        # Assign the most probable topic to each document
        most_probable_topics = lda_output.argmax(axis=1)
        
        # Get the top words for each topic
        feature_names = vectorizer.get_feature_names_out()
        top_words = self._get_top_words(lda_model, feature_names)
        
        # Map the most probable topic's words to each document
        topic_words = [', '.join(top_words[topic]) for topic in most_probable_topics]
        
        # Add the assigned topic and topic words to the DataFrame
        df['assigned_topic'] = most_probable_topics
        df['topic_words'] = topic_words
        
        return df
# Seed words format: {topic_index: [list_of_seed_words]}
# seed_words = {0: ['word1', 'word2'], 1: ['word3', 'word4']}

    def seeded_lda_for_windows(self, windows: list, text_column: str, seed_words: dict, n_topics: int = 10, n_iter: int = 1000):
        """
        Fits a seeded LDA model to each window in the input list of windows.
        """
        lda_results = []
        for window in tqdm(windows, desc="Fitting seeded LDA to windows"):
            df = self.seeded_lda_model(window, text_column, seed_words, n_topics, n_iter)
            lda_results.append(df)
        return lda_results    