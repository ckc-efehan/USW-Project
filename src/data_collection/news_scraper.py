"""
Web scraper for collecting articles about 4-day work week from news websites.

This module scrapes articles from:
- Spiegel Online
- Handelsblatt
- Süddeutsche Zeitung
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
import time
import random
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class NewsScraper:
    """Scraper for news websites to collect articles about 4-day work week."""
    
    def __init__(self, output_dir='data/raw'):
        """
        Initialize the scraper.
        
        Args:
            output_dir (str): Directory to save the scraped data
        """
        self.output_dir = output_dir
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
    def scrape_spiegel(self, search_term='4-Tage-Woche', max_pages=5):
        """
        Scrape articles from Spiegel Online.
        
        Args:
            search_term (str): Term to search for
            max_pages (int): Maximum number of pages to scrape
            
        Returns:
            pd.DataFrame: DataFrame with columns [id, source, url, title, text, date]
        """
        logger.info(f"Scraping Spiegel Online for '{search_term}'")
        articles = []
        
        for page in range(1, max_pages + 1):
            try:
                # Construct search URL
                url = f"https://www.spiegel.de/suche/?suchbegriff={search_term}&seite={page}"
                
                logger.info(f"Scraping page {page} from Spiegel Online")
                response = requests.get(url, headers=self.headers)
                
                if response.status_code != 200:
                    logger.warning(f"Failed to fetch page {page} from Spiegel Online. Status code: {response.status_code}")
                    continue
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract article links (this is a placeholder - actual implementation would need to inspect the website structure)
                article_links = soup.select('article a.href')  # Adjust selector based on actual website structure
                
                for link in article_links:
                    article_url = link.get('href')
                    if not article_url.startswith('http'):
                        article_url = 'https://www.spiegel.de' + article_url
                    
                    # Fetch article content
                    article_data = self._fetch_article_content(article_url, 'spiegel')
                    if article_data:
                        articles.append(article_data)
                
                # Be nice to the server
                time.sleep(random.uniform(1, 3))
                
            except Exception as e:
                logger.error(f"Error scraping page {page} from Spiegel Online: {str(e)}")
        
        # Convert to DataFrame
        df = pd.DataFrame(articles)
        
        # Save to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(self.output_dir, f'spiegel_articles_{timestamp}.csv')
        df.to_csv(output_file, index=False)
        
        logger.info(f"Scraped {len(df)} articles from Spiegel Online. Saved to {output_file}")
        
        return df
    
    def scrape_handelsblatt(self, search_term='4-Tage-Woche', max_pages=5):
        """
        Scrape articles from Handelsblatt.
        
        Args:
            search_term (str): Term to search for
            max_pages (int): Maximum number of pages to scrape
            
        Returns:
            pd.DataFrame: DataFrame with columns [id, source, url, title, text, date]
        """
        logger.info(f"Scraping Handelsblatt for '{search_term}'")
        # Implementation similar to scrape_spiegel but adapted for Handelsblatt
        # This is a placeholder - actual implementation would need to inspect the website structure
        
        return pd.DataFrame()  # Return empty DataFrame for now
    
    def scrape_sueddeutsche(self, search_term='4-Tage-Woche', max_pages=5):
        """
        Scrape articles from Süddeutsche Zeitung.
        
        Args:
            search_term (str): Term to search for
            max_pages (int): Maximum number of pages to scrape
            
        Returns:
            pd.DataFrame: DataFrame with columns [id, source, url, title, text, date]
        """
        logger.info(f"Scraping Süddeutsche Zeitung for '{search_term}'")
        # Implementation similar to scrape_spiegel but adapted for Süddeutsche Zeitung
        # This is a placeholder - actual implementation would need to inspect the website structure
        
        return pd.DataFrame()  # Return empty DataFrame for now
    
    def _fetch_article_content(self, url, source):
        """
        Fetch and parse content from an article URL.
        
        Args:
            url (str): URL of the article
            source (str): Source website name
            
        Returns:
            dict: Article data with keys [id, source, url, title, text, date]
        """
        try:
            response = requests.get(url, headers=self.headers)
            
            if response.status_code != 200:
                logger.warning(f"Failed to fetch article from {url}. Status code: {response.status_code}")
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract article content (this is a placeholder - actual implementation would need to inspect the website structure)
            title = soup.select_one('h1').text.strip() if soup.select_one('h1') else 'No title'
            
            # Extract article text - this would need to be adapted based on the website's structure
            paragraphs = soup.select('article p')  # Adjust selector based on actual website structure
            text = ' '.join([p.text.strip() for p in paragraphs])
            
            # Extract date - this would need to be adapted based on the website's structure
            date_element = soup.select_one('time')  # Adjust selector based on actual website structure
            date = date_element.get('datetime') if date_element else datetime.now().isoformat()
            
            return {
                'id': hash(url),
                'source': source,
                'url': url,
                'title': title,
                'text': text,
                'date': date
            }
            
        except Exception as e:
            logger.error(f"Error fetching article content from {url}: {str(e)}")
            return None
    
    def scrape_all(self, search_term='4-Tage-Woche', max_pages=5):
        """
        Scrape articles from all supported news websites.
        
        Args:
            search_term (str): Term to search for
            max_pages (int): Maximum number of pages to scrape per website
            
        Returns:
            pd.DataFrame: Combined DataFrame with all scraped articles
        """
        logger.info(f"Scraping all news websites for '{search_term}'")
        
        # Scrape from each source
        df_spiegel = self.scrape_spiegel(search_term, max_pages)
        df_handelsblatt = self.scrape_handelsblatt(search_term, max_pages)
        df_sueddeutsche = self.scrape_sueddeutsche(search_term, max_pages)
        
        # Combine results
        df_combined = pd.concat([df_spiegel, df_handelsblatt, df_sueddeutsche], ignore_index=True)
        
        # Save combined results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(self.output_dir, f'all_news_articles_{timestamp}.csv')
        df_combined.to_csv(output_file, index=False)
        
        logger.info(f"Scraped a total of {len(df_combined)} articles from all sources. Saved to {output_file}")
        
        return df_combined

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"news_scraper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    
    # Run scraper
    scraper = NewsScraper()
    df = scraper.scrape_all(search_term='4-Tage-Woche', max_pages=2)
    print(f"Scraped {len(df)} articles in total")