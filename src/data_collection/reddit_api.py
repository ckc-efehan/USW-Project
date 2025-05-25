"""
Reddit API client for collecting posts and comments about 4-day work week.

This module uses the PRAW (Python Reddit API Wrapper) library to interact with the Reddit API
and collect posts and comments from r/de and other relevant subreddits.
"""

import praw
import pandas as pd
import logging
import time
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class RedditCollector:
    """Collector for Reddit posts and comments about 4-day work week."""
    
    def __init__(self, client_id, client_secret, user_agent, output_dir='data/raw'):
        """
        Initialize the Reddit collector.
        
        Args:
            client_id (str): Reddit API client ID
            client_secret (str): Reddit API client secret
            user_agent (str): User agent string for Reddit API
            output_dir (str): Directory to save the collected data
        """
        self.output_dir = output_dir
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def collect_posts(self, subreddit_name='de', search_terms=None, limit=100):
        """
        Collect posts from a subreddit that match the search terms.
        
        Args:
            subreddit_name (str): Name of the subreddit to collect from
            search_terms (list): List of search terms to look for
            limit (int): Maximum number of posts to collect
            
        Returns:
            pd.DataFrame: DataFrame with collected posts
        """
        if search_terms is None:
            search_terms = ['4-Tage-Woche', '4 Tage Woche', 'Arbeitszeitverkürzung', 'Arbeitszeit']
        
        logger.info(f"Collecting posts from r/{subreddit_name} with search terms: {search_terms}")
        
        subreddit = self.reddit.subreddit(subreddit_name)
        posts = []
        
        for term in search_terms:
            logger.info(f"Searching for term: {term}")
            
            try:
                # Search for posts containing the term
                for submission in subreddit.search(term, limit=limit):
                    post_data = {
                        'id': submission.id,
                        'source': f'reddit/r/{subreddit_name}',
                        'url': f'https://www.reddit.com{submission.permalink}',
                        'title': submission.title,
                        'text': submission.selftext,
                        'author': str(submission.author),
                        'score': submission.score,
                        'created_utc': datetime.fromtimestamp(submission.created_utc).isoformat(),
                        'num_comments': submission.num_comments
                    }
                    posts.append(post_data)
                    
                    # Be nice to the API
                    time.sleep(0.5)
            
            except Exception as e:
                logger.error(f"Error collecting posts for term '{term}': {str(e)}")
        
        # Convert to DataFrame
        df = pd.DataFrame(posts)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['id'])
        
        # Save to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(self.output_dir, f'reddit_posts_{subreddit_name}_{timestamp}.csv')
        df.to_csv(output_file, index=False)
        
        logger.info(f"Collected {len(df)} posts from r/{subreddit_name}. Saved to {output_file}")
        
        return df
    
    def collect_comments(self, post_ids, limit=None):
        """
        Collect comments from posts.
        
        Args:
            post_ids (list): List of post IDs to collect comments from
            limit (int): Maximum number of comments to collect per post
            
        Returns:
            pd.DataFrame: DataFrame with collected comments
        """
        logger.info(f"Collecting comments from {len(post_ids)} posts")
        
        comments = []
        
        for post_id in post_ids:
            try:
                submission = self.reddit.submission(id=post_id)
                
                # Replace MoreComments objects with actual comments
                submission.comments.replace_more(limit=limit)
                
                # Collect all comments
                for comment in submission.comments.list():
                    comment_data = {
                        'id': comment.id,
                        'post_id': post_id,
                        'source': 'reddit_comment',
                        'text': comment.body,
                        'author': str(comment.author),
                        'score': comment.score,
                        'created_utc': datetime.fromtimestamp(comment.created_utc).isoformat(),
                        'parent_id': comment.parent_id
                    }
                    comments.append(comment_data)
                
                # Be nice to the API
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error collecting comments for post {post_id}: {str(e)}")
        
        # Convert to DataFrame
        df = pd.DataFrame(comments)
        
        # Save to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(self.output_dir, f'reddit_comments_{timestamp}.csv')
        df.to_csv(output_file, index=False)
        
        logger.info(f"Collected {len(df)} comments. Saved to {output_file}")
        
        return df
    
    def collect_all(self, subreddits=None, search_terms=None, post_limit=100, comment_limit=None):
        """
        Collect posts and comments from multiple subreddits.
        
        Args:
            subreddits (list): List of subreddit names to collect from
            search_terms (list): List of search terms to look for
            post_limit (int): Maximum number of posts to collect per subreddit and search term
            comment_limit (int): Maximum number of comments to collect per post
            
        Returns:
            tuple: (posts_df, comments_df) DataFrames with collected posts and comments
        """
        if subreddits is None:
            subreddits = ['de', 'arbeitsleben', 'Finanzen']
        
        if search_terms is None:
            search_terms = ['4-Tage-Woche', '4 Tage Woche', 'Arbeitszeitverkürzung', 'Arbeitszeit']
        
        logger.info(f"Collecting data from subreddits: {subreddits} with search terms: {search_terms}")
        
        all_posts = []
        
        # Collect posts from each subreddit
        for subreddit in subreddits:
            posts_df = self.collect_posts(subreddit, search_terms, post_limit)
            all_posts.append(posts_df)
        
        # Combine all posts
        combined_posts = pd.concat(all_posts, ignore_index=True)
        
        # Collect comments for all posts
        comments_df = self.collect_comments(combined_posts['id'].tolist(), comment_limit)
        
        # Save combined posts
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(self.output_dir, f'reddit_all_posts_{timestamp}.csv')
        combined_posts.to_csv(output_file, index=False)
        
        logger.info(f"Collected a total of {len(combined_posts)} posts and {len(comments_df)} comments")
        
        return combined_posts, comments_df

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"reddit_collector_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )

    # You would need to provide your own Reddit API credentials
    client_id = "YOUR_CLIENT_ID"
    client_secret = "YOUR_CLIENT_SECRET"
    user_agent = "4-day-work-week-analyzer/1.0 (by /u/YOUR_USERNAME)"
    
    # Run collector
    collector = RedditCollector(client_id, client_secret, user_agent)
    posts_df, comments_df = collector.collect_all(
        subreddits=['de', 'arbeitsleben'],
        search_terms=['4-Tage-Woche', 'Arbeitszeitverkürzung'],
        post_limit=50
    )
    
    print(f"Collected {len(posts_df)} posts and {len(comments_df)} comments in total")