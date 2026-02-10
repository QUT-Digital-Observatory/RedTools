 #emo intensity over time

import pandas as pd
import plotly.express as px
from tqdm.auto import tqdm
from sentence_splitter import SentenceSplitter
import re
from nltk.corpus import stopwords




lexicon_filepath = "data/NRC-Emotion-Intensity-Lexicon-v1.txt"

class EmoIntensityOverTime:
    def __init__(self):
          pass

    def load_lexicon(self, filepath: str) -> pd.DataFrame:
        """
        Load data from a text file into a DataFrame.
        """
        # Read the file using pandas read_csv function, specifying the separator and column names
        lexicon = pd.read_csv(filepath, sep='\t', header=None, names=['Word', 'Emotion', 'Score'])
        return lexicon

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

    def analyse_sentences(self, df_sentences: pd.DataFrame, lexicon: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Analyse the sentiments of multiple sentences using a given lexicon.
        """
        # Ensure the lexicon word column is in lower case for matching
        lexicon['Word'] = lexicon['Word'].str.lower()
        
        # Initialize an empty list to store the results
        results = []
        
        # Define a list of possible emotions (extracted from your lexicon or predefined)
        emotions = lexicon['Emotion'].unique().tolist()

        # Iterate over each sentence in the DataFrame
        for index, row in df_sentences.iterrows():
            # Tokenize the sentence into lower case words
            words = row[text_column].lower().split()
            
            # Filter the lexicon to only include words found in the current sentence
            matched_lexicon = lexicon[lexicon['Word'].isin(words)]
            
            # Summarize the scores by emotion for the words in the sentence
            emotion_scores = matched_lexicon.groupby('Emotion').agg({'Score': 'mean'}).transpose()
            
            # Check if there are any scores to add
            if not emotion_scores.empty:
                result_row = {**row.to_dict(), **emotion_scores.to_dict('records')[0]}
            else:
                # If no scores, initialize scores to 0 for each emotion
                emotion_scores = {emotion: 0 for emotion in emotions}
                result_row = {**row.to_dict(), **emotion_scores}
            
            results.append(result_row)
        
        # Convert results list to DataFrame
        results_df = pd.DataFrame(results)

        # Fill NaN values with 0 for all emotion columns
        for emotion in emotions:
            if emotion in results_df.columns:
                results_df[emotion] = results_df[emotion].fillna(0)
        
        return results_df

    def aggregate_mean_scores(self, df_scores: pd.DataFrame, id_column: str, score_columns: list) -> pd.DataFrame:
        """
        Aggregate sentiment scores by an ID from a DataFrame containing sentiment scores for each sentence.
        """
        # Group by video ID and sum the scores for each specified emotion column
        df_aggregated_scores = df_scores.groupby(id_column)[score_columns].mean().reset_index()

        # Handle metadata, assuming metadata does not include score columns or the video ID column
        metadata_columns = [col for col in df_scores.columns if col not in score_columns + [id_column]]
        df_metadata = df_scores.groupby(id_column)[metadata_columns].agg('first').reset_index()

        # Merge the aggregated scores with the metadata
        df_aggregated = pd.merge(df_aggregated_scores, df_metadata, on=id_column, how='left')

        return df_aggregated
    
    def plot_intensity_over_time(self, data: pd.DataFrame, intensity_columns: list, date_column: str, title: str, freq: str = 'D'):
        """
        Plot the average intensity scores over time for each emotion.
    
        Args:
        data (pd.DataFrame): DataFrame containing the data.
        intensity_columns (list): List of columns containing intensity scores.
        date_column (str): Column name for the dates.
        title (str): Title of the plot.
        freq (str): Frequency for resampling ('D' for daily, 'W' for weekly, 'M' for monthly).
        """
    # Convert the date_column to datetime format if it isn't already
        data[date_column] = pd.to_datetime(data[date_column])

    # Resample the data by the specified frequency and calculate the mean intensity for each period
        daily_data = data.resample(freq, on=date_column)[intensity_columns].mean().reset_index()

    # Melt the DataFrame to long format
        data_melted = daily_data.melt(id_vars=[date_column], value_vars=intensity_columns, var_name='Emotion', value_name='Intensity')

    # Create a line plot using Plotly Express
        fig = px.line(data_melted, x=date_column, y='Intensity', color='Emotion', title=title)
    
    # Update layout to improve readability
        fig.update_layout(xaxis_title='Date', yaxis_title='Average Intensity', xaxis_tickangle=-45)

        return fig