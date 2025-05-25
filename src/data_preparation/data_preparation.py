"""
Data preparation module for 4-day work week argument analysis.

This module handles:
1. Cleaning text data (removing HTML tags, emojis, irrelevant content)
2. Splitting texts into individual sentences
3. Creating a structured dataset for analysis
4. Preparing data for manual labeling
"""

import pandas as pd
import re
import os
import logging
import nltk
from bs4 import BeautifulSoup
import emoji
from datetime import datetime
import uuid

# Download necessary NLTK resources
try:
    nltk.download('punkt', quiet=True)
except:
    pass

logger = logging.getLogger(__name__)

class DataPreparation:
    """Data preparation for 4-day work week argument analysis."""
    
    def __init__(self, input_dir='data/raw', output_dir='data/processed'):
        """
        Initialize the data preparation module.
        
        Args:
            input_dir (str): Directory containing raw data files
            output_dir (str): Directory to save processed data
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def clean_text(self, text):
        """
        Clean text by removing HTML tags, emojis, and other irrelevant content.
        
        Args:
            text (str): Text to clean
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        text = BeautifulSoup(text, 'html.parser').get_text()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove emojis
        text = emoji.replace_emoji(text, replace='')
        
        # Remove special characters and extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)]', '', text)
        text = text.strip()
        
        return text
    
    def split_into_sentences(self, text):
        """
        Split text into individual sentences.
        
        Args:
            text (str): Text to split
            
        Returns:
            list: List of sentences
        """
        if not isinstance(text, str) or not text:
            return []
        
        # Use NLTK's sentence tokenizer
        sentences = nltk.sent_tokenize(text, language='german')
        
        # Clean each sentence
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def process_file(self, file_path):
        """
        Process a single data file.
        
        Args:
            file_path (str): Path to the data file
            
        Returns:
            tuple: (original_df, sentences_df) DataFrames with original and processed data
        """
        logger.info(f"Processing file: {file_path}")
        
        try:
            # Read the data file
            df = pd.read_csv(file_path)
            
            # Ensure required columns exist
            required_columns = ['id', 'source', 'text']
            if not all(col in df.columns for col in required_columns):
                logger.warning(f"File {file_path} is missing required columns: {required_columns}")
                return None, None
            
            # Clean text data
            df['cleaned_text'] = df['text'].apply(self.clean_text)
            
            # Create sentences DataFrame
            sentences_data = []
            
            for _, row in df.iterrows():
                text_id = row['id']
                source = row['source']
                sentences = self.split_into_sentences(row['cleaned_text'])
                
                for i, sentence in enumerate(sentences):
                    sentence_data = {
                        'id': str(uuid.uuid4()),
                        'source': source,
                        'text_id': text_id,
                        'sentence_index': i,
                        'sentence': sentence
                    }
                    sentences_data.append(sentence_data)
            
            sentences_df = pd.DataFrame(sentences_data)
            
            return df, sentences_df
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return None, None
    
    def process_all_files(self):
        """
        Process all data files in the input directory.
        
        Returns:
            tuple: (original_df, sentences_df) Combined DataFrames with all processed data
        """
        logger.info(f"Processing all files in {self.input_dir}")
        
        all_original_dfs = []
        all_sentences_dfs = []
        
        # Process each CSV file in the input directory
        for file_name in os.listdir(self.input_dir):
            if file_name.endswith('.csv'):
                file_path = os.path.join(self.input_dir, file_name)
                original_df, sentences_df = self.process_file(file_path)
                
                if original_df is not None and sentences_df is not None:
                    all_original_dfs.append(original_df)
                    all_sentences_dfs.append(sentences_df)
        
        # Combine all DataFrames
        if all_original_dfs and all_sentences_dfs:
            combined_original_df = pd.concat(all_original_dfs, ignore_index=True)
            combined_sentences_df = pd.concat(all_sentences_dfs, ignore_index=True)
            
            # Save processed data
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            original_output_file = os.path.join(self.output_dir, f'original_data_{timestamp}.csv')
            sentences_output_file = os.path.join(self.output_dir, f'sentences_data_{timestamp}.csv')
            
            combined_original_df.to_csv(original_output_file, index=False)
            combined_sentences_df.to_csv(sentences_output_file, index=False)
            
            logger.info(f"Processed {len(combined_original_df)} original texts and extracted {len(combined_sentences_df)} sentences")
            logger.info(f"Saved processed data to {original_output_file} and {sentences_output_file}")
            
            return combined_original_df, combined_sentences_df
        else:
            logger.warning("No data was processed")
            return pd.DataFrame(), pd.DataFrame()
    
    def prepare_for_labeling(self, sentences_df, sample_size=50):
        """
        Prepare a sample of sentences for manual labeling.
        
        Args:
            sentences_df (pd.DataFrame): DataFrame with sentences
            sample_size (int): Number of sentences to sample for labeling
            
        Returns:
            pd.DataFrame: DataFrame with sampled sentences for labeling
        """
        logger.info(f"Preparing {sample_size} sentences for manual labeling")
        
        # Sample sentences for labeling
        if len(sentences_df) > sample_size:
            labeling_df = sentences_df.sample(sample_size, random_state=42)
        else:
            labeling_df = sentences_df.copy()
        
        # Add label column
        labeling_df['label'] = None
        
        # Save labeling file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        labeling_file = os.path.join(self.output_dir, f'labeling_sample_{timestamp}.csv')
        labeling_df.to_csv(labeling_file, index=False)
        
        logger.info(f"Prepared {len(labeling_df)} sentences for labeling. Saved to {labeling_file}")
        
        return labeling_df

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"data_preparation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    
    # Run data preparation
    data_prep = DataPreparation()
    original_df, sentences_df = data_prep.process_all_files()
    
    if not sentences_df.empty:
        labeling_df = data_prep.prepare_for_labeling(sentences_df)
        print(f"Processed {len(original_df)} original texts and extracted {len(sentences_df)} sentences")
        print(f"Prepared {len(labeling_df)} sentences for manual labeling")