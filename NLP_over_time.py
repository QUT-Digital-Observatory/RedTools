#NLP over Time

import pandas as pd
import numpy as np
from collections import Counter
from nltk import ngrams
import spacy
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from utils import (
    expand_dataframe_with_sentences,
    processed_text_column,
    remove_stopwords,
    get_frames,
)

class NLP_over_time:
    def __init__(self):
        pass

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

    def top_n_terms_per_frame(self, frames: list, text_column: str, n: int = 10) -> list:
        """
        Extracts the top N terms from each data frame in the list of frames based on the specified text column and assigns a count to each row.

        :param frames: List of data frames.
        :param text_column: The column containing text data.
        :param n: Number of top terms to extract.
        :return: List of data frames with term counts.
        """
        top_terms = []

        for frame in tqdm(frames, desc="Extracting top terms"):
            # Combine text from all rows in the frame
            combined_text = ' '.join(frame[text_column].tolist())

            # Tokenize the text
            tokens = combined_text.split()

            # Count the frequency of each token
            term_counts = Counter(tokens)

            # Get the top N terms
            top_n_terms = [term for term, _ in term_counts.most_common(n)]

            # Assign counts of top N terms to each row in the frame
            for term in top_n_terms:
                frame[term] = frame[text_column].apply(lambda x: x.split().count(term))

            top_terms.append(frame)

        return top_terms

    def top_n_grams_per_frame(self, frames: list, text_column: str, n: int = 10, ngram_range: int = 2) -> list:
        """
        Extracts the top N n-grams from each data frame in the list of frames based on the specified text column and n-gram range.
        """
        top_ngrams = []
        
        for frame in tqdm(frames, desc="Extracting top n-grams"):
            # Combine text from all rows in the frame
            combined_text = ' '.join(frame[text_column].tolist())

            # Tokenize the text
            tokens = combined_text.split()

            # Generate n-grams
            ngrams_list = list(ngrams(tokens, ngram_range))

            # Count the frequency of each n-gram
            ngram_counts = Counter(ngrams_list)

            # Get the top N n-grams
            top_ngrams.append(ngram_counts.most_common(n))
        
        return top_ngrams 

    def get_named_entities_per_frame(self, frames: list, text_column: str) -> list:
        """
        Extracts named entities from the text in the specified column for each data frame in the list of frames.
        """
        named_entities = []

        nlp = spacy.load("en_core_web_trf")
        for frame in tqdm(frames, desc="Extracting named entities"):
            # Combine text from all rows in the frame
            combined_text = ' '.join(frame[text_column].tolist())

            # Process the text
            doc = nlp(combined_text)
            
            # Extract named entities
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            
            named_entities.append(entities)
        
        return named_entities

