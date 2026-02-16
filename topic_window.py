#voice_tot

import numpy as np
import pandas as pd
from bertopic import BERTopic
import yaml
from datetime import datetime
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import plotly.express as px
import plotly.graph_objects as go
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from utils import (
    load_config,
    expand_dataframe_with_sentences,
    processed_text_column,
    remove_stopwords,
    get_frames,
    create_umap_hdbscan_models,
)

config_path = 'config.yaml'

config = load_config(config_path)
hardware = config.get('hardware', 'GPU')

class TopicWindow:
    def __init__(self, config_path: str):
        self.data = []  # Changed to a list to keep track of weekly data slices
        self.models = []
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def expand_dataframe_with_sentences(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Expands a DataFrame by splitting text into sentences and preprocessing them."""
        return expand_dataframe_with_sentences(df, text_column)

    def processed_text_column(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Preprocesses the text in the specified column of the input DataFrame."""
        return processed_text_column(df, text_column)

    def remove_stopwords(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Removes stopwords from the text in the specified column of the input DataFrame."""
        return remove_stopwords(df, text_column)

    def get_frames(self, data: pd.DataFrame, date_column: str, timescale: str = 'week') -> list:
        """Splits the DataFrame into intervals based on the specified timescale."""
        frames = get_frames(data, date_column, timescale)
        self.data = frames
        return frames

    def fit(self, data: pd.DataFrame, date_column: str, text_column: str, timescale: str = 'week'):
        """
        Fits a BERTopic model to each weekly interval in the input data.
        """
        umap_model, hdbscan_model = create_umap_hdbscan_models(self.config, hardware)
        bertopic_params = self.config['bertopic']

        frames = self.get_frames(data, date_column, timescale)
        for frame in tqdm(frames, desc="Fitting BERTopic to frames"):
            if not frame.empty:
                model = BERTopic(
                    umap_model=umap_model,
                    hdbscan_model=hdbscan_model,
                    calculate_probabilities=bertopic_params['calculate_probabilities'],
                    verbose=bertopic_params['verbose'],
                    min_topic_size=bertopic_params['min_topic_size']
                ).fit(frame[text_column])
                self.models.append(model)

    def windows_merge_topics(self, max_window_size: int = 4) -> tuple:
        """
        Merges topics from multiple BERTopic models and corresponding weekly data based on a sliding window of specified size,
        with a progressive start and tapering end around the core 4-week window.
        """
        merged_topics = []
        merged_data_frames = []
        total_models = len(self.models)
        
        # Handle both model merging and data frame merging in progressive, full, and tapering phases
        for window_size in range(1, max_window_size + 1):
            if window_size <= total_models:
                models_window = self.models[:window_size]
                data_window = pd.concat(self.data[:window_size]).reset_index(drop=True)
                merged_model = BERTopic.merge_models(models_window)
                merged_topics.append(merged_model)
                merged_data_frames.append(data_window)

        for i in range(1, total_models - max_window_size + 1):
            models_window = self.models[i:i + max_window_size]
            data_window = pd.concat(self.data[i:i + max_window_size]).reset_index(drop=True)
            merged_model = BERTopic.merge_models(models_window)
            merged_topics.append(merged_model)
            merged_data_frames.append(data_window)

        for window_size in range(max_window_size - 1, 0, -1):
            if total_models - window_size >= 0:
                models_window = self.models[-window_size:]
                data_window = pd.concat(self.data[-window_size:]).reset_index(drop=True)
                merged_model = BERTopic.merge_models(models_window)
                merged_topics.append(merged_model)
                merged_data_frames.append(data_window)

        return merged_topics, merged_data_frames

    def get_docs(self, merged_data_frames, merged_topics, text_column: str) -> list:
        """
        Assigns topics to the documents in the merged DataFrames based on the merged BERTopic models.
        """
        docs_with_topics = []
        for idx, (model, data) in enumerate(zip(merged_topics, merged_data_frames)):
            try:
                text_data = data[text_column].tolist()
                docs = model.get_document_info(text_data, df=data)
                docs_with_topics.append(docs)
            except Exception as e:
                print(f"Error processing model {idx}: {e}")
                continue
        
        return docs_with_topics    
    
    def topics_over_time(self, docs_with_topics: list, time_column: str = 'created_utc', timescale: str = 'week') -> list:
        """
        Calculates the number of documents per topic over time based on the specified timescale.
        """
        topics_over_time = []
        for data in docs_with_topics:
            data = data[data['Topic'] != -1]

            # Ensure the time column is in datetime format
            if pd.api.types.is_numeric_dtype(data[time_column]):
                data[time_column] = pd.to_datetime(data[time_column], unit='s')
            else:
                data[time_column] = pd.to_datetime(data[time_column])

            # Resample the data based on the specified timescale
            if timescale == 'hour':
                data['time_period'] = data[time_column].dt.to_period('H')
            elif timescale == 'day':
                data['time_period'] = data[time_column].dt.to_period('D')
            elif timescale == 'week':
                data['time_period'] = data[time_column].dt.to_period('W')
            elif timescale == 'month':
                data['time_period'] = data[time_column].dt.to_period('M')
            elif timescale == 'year':
                data['time_period'] = data[time_column].dt.to_period('Y')
            else:
                raise ValueError("Invalid timescale. Choose from 'hour', 'day', 'week', 'month', 'year'.")

            # Group by time_period and Topic, then count the occurrences
            topic_counts = data.groupby(['time_period', 'Topic']).size().unstack().fillna(0)
            topics_over_time.append(topic_counts)

        return topics_over_time
    
    def plot_topics_over_time(self, topics_over_time: list) -> list:
        """
        Plots the number of documents per topic over time for each time period.
        """
        figs = []
        for i, topic_counts in enumerate(topics_over_time):
            topic_counts = topic_counts.reset_index()
            topic_counts['time_period'] = topic_counts['time_period'].astype(str)  # Convert Period to string
            topic_counts_melted = topic_counts.melt(id_vars=['time_period'], var_name='Topic', value_name='Number of Documents')
            fig = px.line(topic_counts_melted, x='time_period', y='Number of Documents', color='Topic', title=f"Topics Over Time (Window {i+1})", labels={"value": "Number of Documents", "time_period": "Time Period"})
            figs.append(fig)
        return figs

    def plot_intertopic_distances(self, merged_topics: list):
        """
        Plot the intertopic distance for a specific window index.
        """
        figs = []
        for model in (merged_topics):
            fig = model.visualize_topics()
            figs.append(fig)
        return figs
    
    def plot_line_flow(self, topics_over_time: list, top_n: int = 20):
        """
        Plot the rank flow for a specific window index.

        :param topics_over_time: List of DataFrames with 'date' as index and topics as columns
        :param top_n: Number of top topics to display by rank
        :return: List of Plotly figure objects
        """
        figs = []
        
        for i, topic_counts in enumerate(topics_over_time):
            # Ensure the data is sorted by date for proper plotting
            topic_counts = topic_counts.sort_index()
            
            # Convert PeriodIndex to datetime if necessary
            if isinstance(topic_counts.index, pd.PeriodIndex):
                topic_counts.index = topic_counts.index.to_timestamp()
            
            # Compute the total counts for each topic over the time period
            total_counts = topic_counts.sum(axis=0)
            
            # Select the top N topics by total counts
            top_topics = total_counts.nlargest(top_n).index
            filtered_topic_counts = topic_counts[top_topics]
            
            # Compute the rank of each topic for each time period
            ranks = filtered_topic_counts.rank(axis=1, method='first', ascending=False)
            
            # Convert the ranks DataFrame to long format for Plotly
            ranks_long = ranks.reset_index().melt(id_vars='time_period', var_name='topic', value_name='rank')
            
            # Create the line plot using Plotly
            fig = px.line(
                ranks_long,
                x='time_period',
                y='rank',
                color='topic',
                title=f"Rank Flow Over Time (Window {i+1})",
                labels={"rank": "Rank", "time_period": "Date"}
            )
            fig.update_yaxes(autorange="reversed")  # Ensure the top rank is at the top of the graph
            figs.append(fig)
        
        return figs

    def plot_stacked_area(self, topics_over_time: list, top_n: int = 20):
        """
        Plot the streamgraph for a specific window index.

        :param topics_over_time: List of DataFrames with 'date' as index and topics as columns
        :param top_n: Number of top topics to display by rank
        :return: List of Plotly figure objects
        """
        figs = []
        
        for i, topic_counts in enumerate(topics_over_time):
            # Ensure the data is sorted by date for proper plotting
            topic_counts = topic_counts.sort_index()
            
            # Convert PeriodIndex to datetime if necessary
            if isinstance(topic_counts.index, pd.PeriodIndex):
                topic_counts.index = topic_counts.index.to_timestamp()
            
            # Compute the total counts for each topic over the time period
            total_counts = topic_counts.sum(axis=0)
            
            # Select the top N topics by total counts
            top_topics = total_counts.nlargest(top_n).index
            filtered_topic_counts = topic_counts[top_topics]
            
            # Convert the DataFrame to long format for Plotly
            topic_counts_long = filtered_topic_counts.reset_index().melt(id_vars='time_period', var_name='topic', value_name='count')
            
            # Create the streamgraph using Plotly
            fig = px.area(
                topic_counts_long,
                x='time_period',
                y='count',
                color='topic',
                title=f"Streamgraph of Top {top_n} Topics Over Time (Window {i+1})",
                labels={"count": "Count", "time_period": "Date"}
            )
            figs.append(fig)
        
        return figs    

    def all_model(self, data, text_column) -> BERTopic:
        """
        Fit a BERTopic model to the input data.
        """
        umap_model, hdbscan_model = create_umap_hdbscan_models(self.config, hardware)
        bertopic_params = self.config['bertopic']

        all_model = BERTopic(umap_model=umap_model,
                         hdbscan_model=hdbscan_model,
                         calculate_probabilities=bertopic_params['calculate_probabilities'],
                        verbose=bertopic_params['verbose'],
                        min_topic_size=bertopic_params['min_topic_size'])
        all_model.fit_transform (data[text_column])
        return all_model
    
    def get_all_model_docs(self, all_model, data, text_column) -> pd.DataFrame:
        """
        Get all documents from the merged model.
        """
        all_docs = all_model.get_document_info(data[text_column].tolist(), df=data)
        return all_docs
    
    def get_topic_list_from_frame(self, merged_topics, window_idx):
        """
        Get the list of topics from a specific window index.
        """
        return merged_topics[window_idx].get_topic_info()

    def get_all_topics_from_frames(self, merged_topics):
        """
        Get all topics from all windows.
        """
        all_topics = []
        for window in merged_topics:
            all_topics.append(window.get_topic_info())
        return all_topics    
    
    def plot_single_topic_over_time(self, tot: list, topic: int, window_idx: int):  
        """
        Plot the number of documents per topic over time for a specific topic.
        """
        topic_counts = tot[window_idx]
        topic_counts = topic_counts[topic].reset_index()
        fig = px.line(topic_counts, x='days', y=topic, title=f"Topic {topic} Over Time (Window {window_idx+1})", labels={"value": "Number of Documents", "days": "Days"})
        return fig


