"""
Modeling module for 4-day work week argument analysis.

This module implements:
1. A rule-based pre-filtering using spaCy
2. A transformer-based classification using a German NLI model
3. A combined pipeline for argument classification
"""

import pandas as pd
import numpy as np
import spacy
import logging
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datetime import datetime

logger = logging.getLogger(__name__)

class ArgumentClassifier:
    """Classifier for arguments about 4-day work week."""
    
    def __init__(self, model_name='svalabs/gbert-large-zeroshot-nli', output_dir='data/processed'):
        """
        Initialize the argument classifier.
        
        Args:
            model_name (str): Name of the Huggingface model to use
            output_dir (str): Directory to save the classification results
        """
        self.output_dir = output_dir
        self.model_name = model_name
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Load spaCy model for rule-based filtering
        try:
            self.nlp = spacy.load('de_core_news_md')
            logger.info("Loaded spaCy model: de_core_news_md")
        except:
            logger.warning("Could not load spaCy model. Downloading de_core_news_sm instead.")
            spacy.cli.download('de_core_news_sm')
            self.nlp = spacy.load('de_core_news_sm')
        
        # Define rule-based triggers
        self.pro_triggers = [
            "produktivität gesteigert", "mehr zufriedenheit", "besseres wohlbefinden",
            "work-life-balance", "weniger stress", "mehr freizeit", "bessere vereinbarkeit",
            "höhere motivation", "weniger burnout", "bessere konzentration"
        ]
        
        self.contra_triggers = [
            "höhere arbeitslast", "schwierigkeit in umsetzung", "ungeeignet",
            "nicht praktikabel", "zu teuer", "wirtschaftlich nicht tragbar",
            "wettbewerbsnachteil", "personalengpässe", "kundenbedürfnisse"
        ]
        
        # Load transformer model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            logger.info(f"Loaded transformer model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading transformer model: {str(e)}")
            self.tokenizer = None
            self.model = None
    
    def rule_based_classification(self, text):
        """
        Apply rule-based classification using spaCy.
        
        Args:
            text (str): Text to classify
            
        Returns:
            str: Classification result ('pro', 'contra', or 'neutral')
        """
        if not isinstance(text, str) or not text:
            return 'neutral'
        
        # Convert to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Check for pro triggers
        for trigger in self.pro_triggers:
            if trigger in text_lower:
                return 'pro'
        
        # Check for contra triggers
        for trigger in self.contra_triggers:
            if trigger in text_lower:
                return 'contra'
        
        # Use spaCy for more advanced pattern matching
        doc = self.nlp(text)
        
        # Check for positive sentiment patterns
        positive_patterns = [
            # Productivity patterns
            [{'LEMMA': 'produktivität'}, {'OP': '*'}, {'LEMMA': 'steigen'}],
            [{'LEMMA': 'produktivität'}, {'OP': '*'}, {'LEMMA': 'erhöhen'}],
            # Satisfaction patterns
            [{'LEMMA': 'zufriedenheit'}, {'OP': '*'}, {'LEMMA': 'steigen'}],
            [{'LEMMA': 'zufrieden'}, {'OP': '*'}, {'LEMMA': 'sein'}],
            # Well-being patterns
            [{'LEMMA': 'wohlbefinden'}, {'OP': '*'}, {'LEMMA': 'verbessern'}],
            [{'LEMMA': 'gesundheit'}, {'OP': '*'}, {'LEMMA': 'verbessern'}]
        ]
        
        # Check for negative sentiment patterns
        negative_patterns = [
            # Cost patterns
            [{'LEMMA': 'kosten'}, {'OP': '*'}, {'LEMMA': 'steigen'}],
            [{'LEMMA': 'teuer'}, {'OP': '*'}, {'LEMMA': 'sein'}],
            # Implementation difficulty patterns
            [{'LEMMA': 'schwierig'}, {'OP': '*'}, {'LEMMA': 'umsetzen'}],
            [{'LEMMA': 'problem'}, {'OP': '*'}, {'LEMMA': 'verursachen'}],
            # Competitive disadvantage patterns
            [{'LEMMA': 'wettbewerb'}, {'OP': '*'}, {'LEMMA': 'nachteil'}],
            [{'LEMMA': 'konkurrenz'}, {'OP': '*'}, {'LEMMA': 'nachteil'}]
        ]
        
        # This is a simplified version - in a real implementation, 
        # we would use spaCy's matcher to check for these patterns
        
        # For now, return neutral if no simple trigger was found
        return 'neutral'
    
    def transformer_classification(self, text):
        """
        Apply transformer-based classification using NLI model.
        
        Args:
            text (str): Text to classify
            
        Returns:
            str: Classification result ('pro', 'contra', or 'neutral')
            float: Confidence score
        """
        if not isinstance(text, str) or not text or self.model is None:
            return 'neutral', 0.0
        
        # Define hypothesis for NLI
        hypotheses = {
            'pro': 'Die 4-Tage-Woche ist eine gute Idee.',
            'contra': 'Die 4-Tage-Woche ist keine gute Idee.',
            'neutral': 'Die 4-Tage-Woche ist weder gut noch schlecht.'
        }
        
        results = {}
        
        for label, hypothesis in hypotheses.items():
            # Prepare input for the model
            inputs = self.tokenizer(text, hypothesis, return_tensors='pt', truncation=True, max_length=512)
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                prediction = torch.nn.functional.softmax(outputs.logits, dim=1)
                
                # Get entailment score (position depends on the model's output format)
                # For NLI models, typically: 0 = contradiction, 1 = neutral, 2 = entailment
                entailment_score = prediction[0, 2].item()
                results[label] = entailment_score
        
        # Get the label with the highest score
        best_label = max(results, key=results.get)
        confidence = results[best_label]
        
        return best_label, confidence
    
    def classify_sentence(self, sentence):
        """
        Classify a sentence using the combined pipeline.
        
        Args:
            sentence (str): Sentence to classify
            
        Returns:
            dict: Classification result with keys [sentence, rule_based, transformer, final, confidence]
        """
        # Step 1: Rule-based pre-filtering
        rule_based_result = self.rule_based_classification(sentence)
        
        # Step 2: Transformer-based classification
        transformer_result, confidence = self.transformer_classification(sentence)
        
        # Step 3: Combine results
        # If rule-based is not neutral, use it with high confidence
        if rule_based_result != 'neutral':
            final_result = rule_based_result
            final_confidence = 0.9  # High confidence for rule-based matches
        else:
            final_result = transformer_result
            final_confidence = confidence
        
        return {
            'sentence': sentence,
            'rule_based': rule_based_result,
            'transformer': transformer_result,
            'final': final_result,
            'confidence': final_confidence
        }
    
    def classify_dataset(self, sentences_df, sentence_column='sentence'):
        """
        Classify all sentences in a dataset.
        
        Args:
            sentences_df (pd.DataFrame): DataFrame with sentences
            sentence_column (str): Name of the column containing sentences
            
        Returns:
            pd.DataFrame: DataFrame with classification results
        """
        logger.info(f"Classifying {len(sentences_df)} sentences")
        
        results = []
        
        for _, row in sentences_df.iterrows():
            sentence = row[sentence_column]
            
            # Get classification result
            result = self.classify_sentence(sentence)
            
            # Add original row data
            for col in sentences_df.columns:
                result[col] = row[col]
            
            results.append(result)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(self.output_dir, f'classification_results_{timestamp}.csv')
        results_df.to_csv(output_file, index=False)
        
        logger.info(f"Classified {len(results_df)} sentences. Saved to {output_file}")
        
        return results_df
    
    def get_statistics(self, results_df):
        """
        Get statistics about the classification results.
        
        Args:
            results_df (pd.DataFrame): DataFrame with classification results
            
        Returns:
            dict: Statistics about the classification results
        """
        stats = {
            'total': len(results_df),
            'pro': len(results_df[results_df['final'] == 'pro']),
            'contra': len(results_df[results_df['final'] == 'contra']),
            'neutral': len(results_df[results_df['final'] == 'neutral']),
            'rule_based_used': len(results_df[results_df['rule_based'] != 'neutral']),
            'transformer_used': len(results_df[results_df['rule_based'] == 'neutral']),
            'avg_confidence': results_df['confidence'].mean()
        }
        
        # Calculate percentages
        stats['pro_percent'] = (stats['pro'] / stats['total']) * 100
        stats['contra_percent'] = (stats['contra'] / stats['total']) * 100
        stats['neutral_percent'] = (stats['neutral'] / stats['total']) * 100
        
        return stats

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"modeling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    
    # Example sentences for testing
    example_sentences = [
        "Studien zeigen, dass Mitarbeiter an vier Tagen mehr leisten.",
        "Viele Branchen können den Arbeitsausfall nicht kompensieren.",
        "Island hat mehrere Pilotprojekte zur 4-Tage-Woche getestet.",
        "Die Produktivität steigt durch ausgeruhte Mitarbeiter.",
        "Die Work-Life-Balance verbessert sich deutlich.",
        "Die Belastung an den Arbeitstagen wird höher sein.",
        "In der Pflege ist eine 4-Tage-Woche kaum umsetzbar."
    ]
    
    # Create test DataFrame
    test_df = pd.DataFrame({'sentence': example_sentences})
    
    # Run classifier
    classifier = ArgumentClassifier()
    results_df = classifier.classify_dataset(test_df)
    
    # Print results
    print("\nClassification Results:")
    for _, row in results_df.iterrows():
        print(f"Sentence: {row['sentence']}")
        print(f"Classification: {row['final']} (Confidence: {row['confidence']:.2f})")
        print(f"Rule-based: {row['rule_based']}, Transformer: {row['transformer']}")
        print("-" * 50)
    
    # Print statistics
    stats = classifier.get_statistics(results_df)
    print("\nStatistics:")
    print(f"Total sentences: {stats['total']}")
    print(f"Pro: {stats['pro']} ({stats['pro_percent']:.1f}%)")
    print(f"Contra: {stats['contra']} ({stats['contra_percent']:.1f}%)")
    print(f"Neutral: {stats['neutral']} ({stats['neutral_percent']:.1f}%)")
    print(f"Average confidence: {stats['avg_confidence']:.2f}")