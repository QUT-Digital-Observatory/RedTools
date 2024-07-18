#voice_tot

from pydoc_data import topics
import numpy as np
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

class TopicWindow:
    def __init__(self, config_path: str):
        self.data = []  # Changed to a list to keep track of weekly data slices
        self.models = []
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

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

    def get_frames(self, data: pd.DataFrame, date_column: str, timescale: str = 'week') -> list:
        """
        Splits the DataFrame into intervals based on the specified timescale and the range of dates in the specified date column.
        """
        # Ensure the date column is in datetime format
        if pd.api.types.is_numeric_dtype(data[date_column]):
            # Assuming Unix timestamps
            data[date_column] = pd.to_datetime(data[date_column], unit='s')
        else:
            data[date_column] = pd.to_datetime(data[date_column])
        
        # Calculate the start and end times
        start_time = data[date_column].min()
        end_time = data[date_column].max()
        
        # Initialize a list to store the data slices
        frames = []
        
        if timescale == 'hour':
            freq = 'H'
        elif timescale == 'day':
            freq = 'D'
        elif timescale == 'week':
            freq = 'W'
        elif timescale == 'month':
            freq = 'M'
        elif timescale == 'year':
            freq = 'Y'
        else:
            raise ValueError("Invalid timescale. Choose from 'hour', 'day', 'week', 'month', 'year'.")

        # Generate time ranges based on the specified frequency
        time_ranges = pd.date_range(start=start_time, end=end_time, freq=freq)

        for start, end in zip(time_ranges[:-1], time_ranges[1:]):
            frame = data[(data[date_column] >= start) & (data[date_column] < end)].reset_index(drop=True)
            frames.append(frame)

        # Store the data slices in the instance variable
        self.data = frames
        return frames

    def fit(self, data: pd.DataFrame, date_column: str, text_column: str, timescale: str = 'week'):
        """
        Fits a BERTopic model to each weekly interval in the input data.
        """
        # Extract UMAP and HDBSCAN hyperparameters from the config
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

        frames = self.get_frames(data, date_column, timescale)
        total_frames = len(frames)
        for i, frame in enumerate(frames, start=1):
            print(f"Fitting model for frame {i} of {total_frames}...")
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
            # Ensure text_column data is in list format for BERTopic
                text_data = data[text_column].tolist()
            
            # Debug: Print the first few items of text_data
                print(f"Model {idx}: First few text data items: {text_data[:5]}")
            
            # Get detailed document information, including topics
                docs = model.get_document_info(text_data, df=data)
            
            # Debug: Print the first few rows of the result
                print(f"Model {idx}: First few rows of document info:\n{docs.head()}")
            
            # Append the resulting DataFrame to the list
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

    def merge_frames_into_windows(self, frames, window_size):
        """
        Merge frames into windows of specified size.
        """
        windows = []
        docs_out = []
        total_frames = len(frames)
        for i in range(0, total_frames - window_size + 1):
            window = pd.concat(frames[i:i + window_size]).reset_index(drop=True)
            docs = window['body'].tolist()
            windows.append(window)
            docs_out.append(docs)
        return windows, docs_out
    
    def model_windows(self, windows, text_column):
        """
        Fit a BERTopic model to each window in the input list.
        """
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
        
        models = []
        total_windows = len(windows)
        for i, window in enumerate(windows, start=1):
            print(f"Fitting model for window {i} of {total_windows}...")
            if not window.empty:
                model = BERTopic(
                    umap_model=umap_model, 
                    hdbscan_model=hdbscan_model, 
                    calculate_probabilities=bertopic_params['calculate_probabilities'], 
                    verbose=bertopic_params['verbose'], 
                    min_topic_size=bertopic_params['min_topic_size']
                ).fit(window[text_column])
                models.append(model)
        return models
    #bugged            
    def visualise_hierarchy(self, models):
        """
        Visualise the hierarchy of a specific topic in the model.
        """
        figs = []
        for model in models:
            fig = model.visualize_hierarchy()
            figs.append(fig)
        return figs
    #this is just not right
    def get_hierarchical_topics(self, models: list, docs: list):
        """
        Get the hierarchical topics for each model.

        :param models: List of models.
        :param docs: List of documents where each entry corresponds to a model.
        :return: List of hierarchical topics for each model.
        """
        hierarchical_topics = []
        
        # Ensure that models and docs have the same length
        if len(models) != len(docs):
            raise ValueError("The number of models and documents must be the same.")

        for model, doc_frame in zip(models, docs):
            doc_list = doc_frame['text'].tolist()
            hierarchical_topics.append(model.get_hierarchy(doc_list))
        
        return hierarchical_topics

        
    

#TODO: vis h trees over time
#fit_transform each frame (merge models does not have the cTF-IDF matrix)
#now get h trees for each frame


