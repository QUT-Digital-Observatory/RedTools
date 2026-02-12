#NLP over Time

import pandas as pd
import numpy as np
from collections import Counter
from sentence_splitter import SentenceSplitter
import re
from nltk.corpus import stopwords
from nltk import ngrams
import spacy
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

class NLP_over_time:
    def __init__(self):
        self.stopwords = set(stopwords.words('english'))

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
        df = df.dropna(subset=[text_column])
        df['sentences'] = df[text_column].apply(self.__sentence_chunker__)
        df_expanded = df.explode('sentences')

        df_expanded = df_expanded[df_expanded['sentences'].str.strip() != '']
        df_expanded['sentences'] = df_expanded['sentences'].apply(self.__pre_process__)
        df_expanded[text_column] = df_expanded['sentences']
        df_expanded = df_expanded.drop(columns=['sentences'])

        df_expanded = df_expanded[df_expanded[text_column].str.strip() != '' ]
        
        df_expanded = df_expanded.reset_index(drop=True)

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
        df[text_column] = df[text_column].apply(lambda x: ' '.join([word for word in x.split() if word not in self.stopwords]))
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
            freq = 'h'
        elif timescale == 'day':
            freq = 'D'
        elif timescale == 'week':
            freq = 'W'
        elif timescale == 'month':
            freq = 'ME'
        elif timescale == 'year':
            freq = 'YE'
        else:
            raise ValueError("Invalid timescale. Choose from 'hour', 'day', 'week', 'month', 'year'.")

        # Generate time ranges based on the specified frequency
        time_ranges = pd.date_range(start=start_time, end=end_time, freq=freq)

        time_pairs = list(zip(time_ranges[:-1], time_ranges[1:]))
        for start, end in tqdm(time_pairs, desc="Splitting into frames"):
            frame = data[(data[date_column] >= start) & (data[date_column] < end)].reset_index(drop=True)
            frames.append(frame)

        # Store the data slices in the instance variable
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

