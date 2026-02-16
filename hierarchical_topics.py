# Hierarchical topics

import numpy as np
import pandas as pd
from bertopic import BERTopic
import yaml
import plotly.express as px
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from topic_window import TopicWindow
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS

from utils import load_config, create_bertopic_model

config_path = 'config.yaml'

config = load_config(config_path)
hardware = config.get('hardware', 'GPU')

class HierarchicalTopics:

    def __init__(self, config_path: str):
        self.models = []
        self.topic_windows = TopicWindow(config_path)
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def _create_model(self):
        return create_bertopic_model(self.config, hardware)

    def get_hierarchical_topics(self, data, text_column, date_column, timescale):

        hierarchical_topics = []
        topic_trees = []
        figs = []
        data_exp = self.topic_windows.expand_dataframe_with_sentences(data.copy(), text_column)
        frames = self.topic_windows.get_frames(data_exp, date_column, timescale)
        for frame in tqdm(frames, desc="Building hierarchical topics"):
            model = self._create_model().fit(frame[text_column])
            docs: list = frame[text_column].tolist()
            h_tops = model.hierarchical_topics(docs)
            hierarchical_topics.append(h_tops)
            t_trees = model.get_topic_tree(h_tops)
            topic_trees.append(t_trees)
            fig = model.visualize_hierarchy()
            figs.append(fig)
        return hierarchical_topics, topic_trees, figs    

    def get_topic_embeddings(self, data, text_column, date_column, timescale):
        topic_embeddings = []
        interval_labels = []
        data_exp = self.topic_windows.expand_dataframe_with_sentences(data.copy(), text_column)
        frames = self.topic_windows.get_frames(data_exp, date_column, timescale)
        for i, frame in enumerate(tqdm(frames, desc="Extracting topic embeddings")):
            model = self._create_model().fit(frame[text_column])
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

    def visualize_topic_clusters(self, reduced_vectors, interval_labels, method='pca'):
        df = pd.DataFrame(reduced_vectors, columns=[f'{method.upper()}1', f'{method.upper()}2'])
        df['interval'] = interval_labels

        fig = px.scatter(df, x=f'{method.upper()}1', y=f'{method.upper()}2', color='interval', text='interval',
                         title=f'Topic Evolution Over Time ({method.upper()})')
        fig.update_traces(textposition='top center')
        fig.update_layout(xaxis_title=f'{method.upper()} Component 1', yaxis_title=f'{method.upper()} Component 2')
        fig.show()


    def compute_topic_positions(self, embeddings):
        """Compute y-positions of topics based on their inter-topic distances."""
        distances = pdist(embeddings, metric='cosine')
        dist_matrix = squareform(distances)

        # Use multidimensional scaling to position topics
        mds = MDS(n_components=1, dissimilarity='precomputed', random_state=42)
        positions = mds.fit_transform(dist_matrix).flatten()

        # Normalize positions to [0, 1] range
        pos_range = positions.max() - positions.min()
        if pos_range == 0:
            positions = np.full_like(positions, 0.5)
        else:
            positions = (positions - positions.min()) / pos_range
        return positions

    def plot_topic_evolution(self, embeddings_list, cutoff_similarity=0.75, all_links=True):
        num_intervals = len(embeddings_list)
        fig, ax = plt.subplots(figsize=(max(12, num_intervals * 3), 10))

        # Compute positions for all time intervals
        all_positions = [self.compute_topic_positions(embeddings) for embeddings in embeddings_list]

        # Plot topics for each time interval
        for i, positions in enumerate(all_positions):
            ax.scatter([i] * len(positions), positions, s=50, label=f'Interval {i+1}')
            for j, pos in enumerate(positions):
                ax.annotate(f'T{i+1}_{j+1}', (i, pos), xytext=(5, 0),
                            textcoords='offset points', fontsize=8, alpha=0.7)

        # Draw links between adjacent intervals
        for i in range(num_intervals - 1):
            embeddings1 = embeddings_list[i]
            embeddings2 = embeddings_list[i+1]
            positions1 = all_positions[i]
            positions2 = all_positions[i+1]

            for j, topic1 in enumerate(embeddings1):
                links = []
                for k, topic2 in enumerate(embeddings2):
                    similarity = 1 - cosine(topic1, topic2)
                    if similarity >= cutoff_similarity:
                        links.append((k, similarity))

                if all_links:
                    for k, sim in links:
                        ax.plot([i, i+1], [positions1[j], positions2[k]], 'k-',
                                alpha=0.3, linewidth=sim)
                elif links:
                    k, sim = max(links, key=lambda x: x[1])
                    ax.plot([i, i+1], [positions1[j], positions2[k]], 'k-',
                            alpha=0.5, linewidth=sim)

        ax.set_xlim(-0.5, num_intervals - 0.5)
        ax.set_xticks(range(num_intervals))
        ax.set_xticklabels([f'Interval {i+1}' for i in range(num_intervals)])
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([])

        ax.set_title('Topic Evolution Across Time Intervals')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.show()

    