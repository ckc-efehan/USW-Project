"""
Twitter API client for collecting tweets about 4-day work week.

This module uses the Tweepy library to interact with the Twitter API
and collect tweets with hashtags like #4TageWoche and #NewWork.
"""

import tweepy
import pandas as pd
import logging
import time
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class TwitterCollector:
    """Collector for tweets about 4-day work week."""

    def __init__(self, api_key, api_secret, access_token, access_token_secret, output_dir='data/raw'):
        """
        Initialize the Twitter collector.

        Args:
            api_key (str): Twitter API key
            api_secret (str): Twitter API secret
            access_token (str): Twitter access token
            access_token_secret (str): Twitter access token secret
            output_dir (str): Directory to save the collected data
        """
        self.output_dir = output_dir

        # Authenticate with Twitter
        auth = tweepy.OAuth1UserHandler(
            api_key, api_secret, access_token, access_token_secret
        )
        self.api = tweepy.API(auth, wait_on_rate_limit=True)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

    def collect_tweets(self, hashtags=None, lang='de', count=100):
        """
        Collect tweets with specific hashtags.

        Args:
            hashtags (list): List of hashtags to search for
            lang (str): Language of tweets to collect
            count (int): Maximum number of tweets to collect per hashtag

        Returns:
            pd.DataFrame: DataFrame with collected tweets
        """
        if hashtags is None:
            hashtags = ['4TageWoche', 'NewWork', 'Arbeitszeitverkürzung', 'WorkLifeBalance']

        logger.info(f"Collecting tweets with hashtags: {hashtags}")

        tweets = []

        for hashtag in hashtags:
            logger.info(f"Searching for hashtag: #{hashtag}")

            try:
                # Search for tweets containing the hashtag
                query = f"#{hashtag} -filter:retweets"

                for tweet in tweepy.Cursor(self.api.search_tweets, q=query, lang=lang, tweet_mode='extended').items(
                        count):
                    tweet_data = {
                        'id': tweet.id_str,
                        'source': 'twitter',
                        'url': f'https://twitter.com/user/status/{tweet.id_str}',
                        'text': tweet.full_text,
                        'author': tweet.user.screen_name,
                        'followers_count': tweet.user.followers_count,
                        'created_at': tweet.created_at.isoformat(),
                        'retweet_count': tweet.retweet_count,
                        'favorite_count': tweet.favorite_count,
                        'hashtags': [h['text'] for h in tweet.entities.get('hashtags', [])]
                    }
                    tweets.append(tweet_data)

                    # Be nice to the API
                    time.sleep(0.5)

            except tweepy.TweepyException as e:
                logger.error(f"Error collecting tweets for hashtag '#{hashtag}': {str(e)}")

        # Convert to DataFrame
        df = pd.DataFrame(tweets)

        # Remove duplicates
        df = df.drop_duplicates(subset=['id'])

        # Save to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(self.output_dir, f'twitter_tweets_{timestamp}.csv')
        df.to_csv(output_file, index=False)

        logger.info(f"Collected {len(df)} tweets. Saved to {output_file}")

        return df

    def collect_user_tweets(self, usernames, count=100):
        """
        Collect tweets from specific users.

        Args:
            usernames (list): List of Twitter usernames to collect tweets from
            count (int): Maximum number of tweets to collect per user

        Returns:
            pd.DataFrame: DataFrame with collected tweets
        """
        logger.info(f"Collecting tweets from users: {usernames}")

        tweets = []

        for username in usernames:
            logger.info(f"Collecting tweets from user: @{username}")

            try:
                # Collect tweets from the user
                for tweet in tweepy.Cursor(self.api.user_timeline, screen_name=username, tweet_mode='extended').items(
                        count):
                    tweet_data = {
                        'id': tweet.id_str,
                        'source': f'twitter/@{username}',
                        'url': f'https://twitter.com/{username}/status/{tweet.id_str}',
                        'text': tweet.full_text,
                        'author': tweet.user.screen_name,
                        'followers_count': tweet.user.followers_count,
                        'created_at': tweet.created_at.isoformat(),
                        'retweet_count': tweet.retweet_count,
                        'favorite_count': tweet.favorite_count,
                        'hashtags': [h['text'] for h in tweet.entities.get('hashtags', [])]
                    }
                    tweets.append(tweet_data)

                    # Be nice to the API
                    time.sleep(0.5)

            except tweepy.TweepyException as e:
                logger.error(f"Error collecting tweets from user '@{username}': {str(e)}")

        # Convert to DataFrame
        df = pd.DataFrame(tweets)

        # Save to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(self.output_dir, f'twitter_user_tweets_{timestamp}.csv')
        df.to_csv(output_file, index=False)

        logger.info(f"Collected {len(df)} tweets from users. Saved to {output_file}")

        return df

    def collect_all(self, hashtags=None, usernames=None, lang='de', count=100):
        """
        Collect tweets from hashtags and users.

        Args:
            hashtags (list): List of hashtags to search for
            usernames (list): List of Twitter usernames to collect tweets from
            lang (str): Language of tweets to collect
            count (int): Maximum number of tweets to collect per hashtag/user

        Returns:
            pd.DataFrame: Combined DataFrame with all collected tweets
        """
        if hashtags is None:
            hashtags = ['4TageWoche', 'NewWork', 'Arbeitszeitverkürzung', 'WorkLifeBalance']

        if usernames is None:
            usernames = ['arbeitsagentur', 'BMAS_Bund', 'verdi', 'dgb', 'iab_news']

        logger.info(f"Collecting tweets from hashtags: {hashtags} and users: {usernames}")

        # Collect tweets from hashtags
        hashtag_tweets = self.collect_tweets(hashtags, lang, count)

        # Collect tweets from users
        user_tweets = self.collect_user_tweets(usernames, count)

        # Combine results
        combined_tweets = pd.concat([hashtag_tweets, user_tweets], ignore_index=True)

        # Remove duplicates
        combined_tweets = combined_tweets.drop_duplicates(subset=['id'])

        # Save combined results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(self.output_dir, f'twitter_all_tweets_{timestamp}.csv')
        combined_tweets.to_csv(output_file, index=False)

        logger.info(f"Collected a total of {len(combined_tweets)} tweets. Saved to {output_file}")

        return combined_tweets


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"twitter_collector_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )

    # You would need to provide your own Twitter API credentials
    api_key = "YOUR_API_KEY"
    api_secret = "YOUR_API_SECRET"
    access_token = "YOUR_ACCESS_TOKEN"
    access_token_secret = "YOUR_ACCESS_TOKEN_SECRET"

    # Run collector
    collector = TwitterCollector(api_key, api_secret, access_token, access_token_secret)
    df = collector.collect_all(
        hashtags=['4TageWoche', 'NewWork'],
        usernames=['arbeitsagentur', 'BMAS_Bund'],
        count=50
    )

    print(f"Collected {len(df)} tweets in total")
