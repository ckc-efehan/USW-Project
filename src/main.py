"""
Main script to run the 4-Day Work Week Argument Analysis pipeline.

This script orchestrates the entire data science pipeline:
1. Data collection from various sources
2. Data preprocessing and preparation
3. Model training and prediction
4. Evaluation of results
"""

import os
import logging
import argparse
import pandas as pd
from datetime import datetime

# Import components
from data_collection.news_scraper import NewsScraper
from data_collection.reddit_api import RedditCollector
from data_collection.twitter_api import TwitterCollector
from data_preparation.data_preparation import DataPreparation
from modeling.modeling import ArgumentClassifier
from evaluation.evaluation import ModelEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def collect_data(args):
    """
    Collect data from various sources.

    Args:
        args: Command-line arguments

    Returns:
        bool: Success status
    """
    logger.info("Step 1: Data Collection")

    try:
        # Collect data from news sites
        if args.collect_news:
            logger.info("Collecting data from news sites")
            scraper = NewsScraper(output_dir=args.raw_data_dir)
            news_df = scraper.scrape_all(search_term=args.search_term, max_pages=args.max_pages)
            logger.info(f"Collected {len(news_df)} articles from news sites")

        # Collect data from Reddit
        if args.collect_reddit and args.reddit_client_id and args.reddit_client_secret:
            logger.info("Collecting data from Reddit")
            reddit_collector = RedditCollector(
                client_id=args.reddit_client_id,
                client_secret=args.reddit_client_secret,
                user_agent="4-day-work-week-analyzer/1.0",
                output_dir=args.raw_data_dir
            )
            reddit_posts_df, reddit_comments_df = reddit_collector.collect_all(
                subreddits=args.reddit_subreddits.split(','),
                search_terms=args.search_term.split(','),
                post_limit=args.max_posts
            )
            logger.info(f"Collected {len(reddit_posts_df)} posts and {len(reddit_comments_df)} comments from Reddit")

        # Collect data from Twitter
        if args.collect_twitter and args.twitter_api_key and args.twitter_api_secret:
            logger.info("Collecting data from Twitter")
            twitter_collector = TwitterCollector(
                api_key=args.twitter_api_key,
                api_secret=args.twitter_api_secret,
                access_token=args.twitter_access_token,
                access_token_secret=args.twitter_access_token_secret,
                output_dir=args.raw_data_dir
            )
            twitter_df = twitter_collector.collect_all(
                hashtags=args.twitter_hashtags.split(','),
                count=args.max_tweets
            )
            logger.info(f"Collected {len(twitter_df)} tweets from Twitter")

        return True

    except Exception as e:
        logger.error(f"Error in data collection: {str(e)}")
        return False

def prepare_data(args):
    """
    Prepare data for modeling.

    Args:
        args: Command-line arguments

    Returns:
        tuple: (original_df, sentences_df, labeling_df) or None if error
    """
    logger.info("Step 2: Data Preparation")

    try:
        # Initialize data preparation
        data_prep = DataPreparation(
            input_dir=args.raw_data_dir,
            output_dir=args.processed_data_dir
        )

        # Process all files
        original_df, sentences_df = data_prep.process_all_files()

        if sentences_df.empty:
            logger.warning("No sentences were extracted during data preparation")
            return None

        # Prepare for labeling if needed
        if args.prepare_labeling:
            labeling_df = data_prep.prepare_for_labeling(sentences_df, sample_size=args.labeling_sample_size)
            logger.info(f"Prepared {len(labeling_df)} sentences for manual labeling")
        else:
            labeling_df = None

        logger.info(f"Processed {len(original_df)} original texts and extracted {len(sentences_df)} sentences")

        return original_df, sentences_df, labeling_df

    except Exception as e:
        logger.error(f"Error in data preparation: {str(e)}")
        return None

def run_modeling(args, sentences_df=None):
    """
    Run the modeling component.

    Args:
        args: Command-line arguments
        sentences_df: DataFrame with sentences to classify

    Returns:
        pd.DataFrame: DataFrame with classification results or None if error
    """
    logger.info("Step 3: Modeling")

    try:
        # Initialize classifier
        classifier = ArgumentClassifier(
            model_name=args.model_name,
            output_dir=args.processed_data_dir
        )

        # If no sentences_df is provided, try to load from file
        if sentences_df is None:
            if args.sentences_file:
                logger.info(f"Loading sentences from {args.sentences_file}")
                sentences_df = pd.read_csv(args.sentences_file)
            else:
                logger.error("No sentences provided for classification")
                return None

        # Classify sentences
        results_df = classifier.classify_dataset(sentences_df, sentence_column=args.sentence_column)

        # Get statistics
        stats = classifier.get_statistics(results_df)

        logger.info(f"Classification results: {stats['pro']} pro, {stats['contra']} contra, {stats['neutral']} neutral")

        return results_df

    except Exception as e:
        logger.error(f"Error in modeling: {str(e)}")
        return None

def evaluate_model(args, results_df=None):
    """
    Evaluate the model.

    Args:
        args: Command-line arguments
        results_df: DataFrame with classification results

    Returns:
        dict: Evaluation metrics or None if error
    """
    logger.info("Step 4: Evaluation")

    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(output_dir=args.processed_data_dir)

        # If no results_df is provided, try to load from file
        if results_df is None:
            if args.results_file:
                logger.info(f"Loading results from {args.results_file}")
                results_df = pd.read_csv(args.results_file)
            else:
                logger.error("No results provided for evaluation")
                return None

        # Check if labeled data is available
        if args.labeled_file:
            logger.info(f"Loading labeled data from {args.labeled_file}")
            labeled_df = pd.read_csv(args.labeled_file)

            # Merge labeled data with results
            merged_df = pd.merge(
                results_df,
                labeled_df[['id', args.label_column]],
                on='id',
                how='inner'
            )

            if len(merged_df) == 0:
                logger.warning("No matching samples found between results and labeled data")
                return None

            # Evaluate model
            metrics = evaluator.evaluate(merged_df, true_label_column=args.label_column, pred_label_column='final')

            # Perform error analysis
            misclassified, error_counts = evaluator.error_analysis(
                merged_df,
                true_label_column=args.label_column,
                pred_label_column='final',
                text_column=args.sentence_column
            )

            # Visualize results
            evaluator.visualize_results(metrics)

            # Save metrics
            evaluator.save_metrics(metrics)

            logger.info(f"Evaluation results: Accuracy = {metrics['accuracy']:.4f}")

            return metrics
        else:
            logger.warning("No labeled data available for evaluation")
            return None

    except Exception as e:
        logger.error(f"Error in evaluation: {str(e)}")
        return None

def main():
    """Run the complete pipeline."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='4-Day Work Week Argument Analysis Pipeline')

    # General arguments
    parser.add_argument('--raw-data-dir', type=str, default='data/raw', help='Directory for raw data')
    parser.add_argument('--processed-data-dir', type=str, default='data/processed', help='Directory for processed data')
    parser.add_argument('--steps', type=str, default='all', help='Pipeline steps to run (comma-separated: collect,prepare,model,evaluate,all)')

    # Data collection arguments
    parser.add_argument('--collect-news', action='store_true', help='Collect data from news sites')
    parser.add_argument('--collect-reddit', action='store_true', help='Collect data from Reddit')
    parser.add_argument('--collect-twitter', action='store_true', help='Collect data from Twitter')
    parser.add_argument('--search-term', type=str, default='4-Tage-Woche', help='Search term for data collection')
    parser.add_argument('--max-pages', type=int, default=5, help='Maximum number of pages to scrape from news sites')
    parser.add_argument('--max-posts', type=int, default=100, help='Maximum number of posts to collect from Reddit')
    parser.add_argument('--max-tweets', type=int, default=100, help='Maximum number of tweets to collect')

    # Reddit API arguments
    parser.add_argument('--reddit-client-id', type=str, help='Reddit API client ID')
    parser.add_argument('--reddit-client-secret', type=str, help='Reddit API client secret')
    parser.add_argument('--reddit-subreddits', type=str, default='de,arbeitsleben,Finanzen', help='Subreddits to collect from (comma-separated)')

    # Twitter API arguments
    parser.add_argument('--twitter-api-key', type=str, help='Twitter API key')
    parser.add_argument('--twitter-api-secret', type=str, help='Twitter API secret')
    parser.add_argument('--twitter-access-token', type=str, help='Twitter access token')
    parser.add_argument('--twitter-access-token-secret', type=str, help='Twitter access token secret')
    parser.add_argument('--twitter-hashtags', type=str, default='4TageWoche,NewWork', help='Twitter hashtags to search for (comma-separated)')

    # Data preparation arguments
    parser.add_argument('--prepare-labeling', action='store_true', help='Prepare sample for manual labeling')
    parser.add_argument('--labeling-sample-size', type=int, default=50, help='Number of sentences to sample for labeling')

    # Modeling arguments
    parser.add_argument('--model-name', type=str, default='svalabs/gbert-large-zeroshot-nli', help='Huggingface model name')
    parser.add_argument('--sentences-file', type=str, help='CSV file with sentences to classify')
    parser.add_argument('--sentence-column', type=str, default='sentence', help='Column name for sentences')

    # Evaluation arguments
    parser.add_argument('--results-file', type=str, help='CSV file with classification results')
    parser.add_argument('--labeled-file', type=str, help='CSV file with labeled data')
    parser.add_argument('--label-column', type=str, default='label', help='Column name for true labels')

    args = parser.parse_args()

    # Create directories if they don't exist
    os.makedirs(args.raw_data_dir, exist_ok=True)
    os.makedirs(args.processed_data_dir, exist_ok=True)

    logger.info("Starting 4-Day Work Week Argument Analysis Pipeline")

    # Determine which steps to run
    steps = args.steps.lower().split(',')
    run_all = 'all' in steps

    # Step 1: Data Collection
    if run_all or 'collect' in steps:
        success = collect_data(args)
        if not success and run_all:
            logger.error("Data collection failed. Stopping pipeline.")
            return

    # Step 2: Data Preparation
    sentences_df = None
    if run_all or 'prepare' in steps:
        result = prepare_data(args)
        if result:
            _, sentences_df, _ = result
        elif run_all:
            logger.error("Data preparation failed. Stopping pipeline.")
            return

    # Step 3: Modeling
    results_df = None
    if run_all or 'model' in steps:
        results_df = run_modeling(args, sentences_df)
        if results_df is None and run_all:
            logger.error("Modeling failed. Stopping pipeline.")
            return

    # Step 4: Evaluation
    if run_all or 'evaluate' in steps:
        metrics = evaluate_model(args, results_df)
        if metrics is None and run_all:
            logger.warning("Evaluation could not be performed. This may be normal if no labeled data is available.")

    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    main()
