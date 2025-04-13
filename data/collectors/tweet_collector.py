import snscrape.modules.twitter as sntwitter
from datetime import datetime, timedelta
import json
import os
import logging
from typing import List, Dict
import time
import pandas as pd
import schedule

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TweetCollector:
    def __init__(self):
        """Initialize tweet collector."""
        # List of influential financial figures to track
        self.influential_figures = {
            'elonmusk': 'Elon Musk',
            'realDonaldTrump': 'Donald Trump',
            'cathiewood': 'Cathie Wood',
            'chamath': 'Chamath Palihapitiya',
            'BillAckman': 'Bill Ackman'
        }
        
        # Market keywords for relevance filtering
        self.market_keywords = [
            # Market terms
            "stock", "earnings", "market", "trading", "investing", "finance",
            # Economic indicators
            "fed", "interest rates", "inflation", "unemployment", "gdp", "economy",
            # Market indices
            "dow", "nasdaq", "s&p", "spx", "nasdaq-100", "russell",
            # Sectors
            "tech", "financial", "healthcare", "energy", "consumer", "industrial",
            # Market events
            "ipo", "merger", "acquisition", "dividend", "split", "bankruptcy",
            # Trading
            "bull", "bear", "rally", "correction", "crash", "volatility",
            # Trading actions
            "buy", "sell", "hold", "long", "short", "position", "exit", "entry",
            # Sentiment indicators
            "bullish", "bearish", "neutral", "overbought", "oversold", "trend",
            # Market analysis
            "technical", "fundamental", "analysis", "chart", "pattern", "support",
            "resistance", "breakout", "breakdown", "momentum", "volume",
            # Financial instruments
            "option", "futures", "etf", "crypto", "bitcoin", "forex", "commodity",
            # Company actions
            "earnings", "guidance", "upgrade", "downgrade", "initiate", "coverage",
            "target", "price target", "rating", "analyst", "report"
        ]
        
        # Load last tweet IDs for each user
        self.last_tweet_ids = self.load_last_tweet_ids()
    
    def load_last_tweet_ids(self) -> Dict[str, str]:
        """Load last tweet IDs from file."""
        try:
            if os.path.exists('last_tweet_ids.json'):
                with open('last_tweet_ids.json', 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading last tweet IDs: {str(e)}")
        return {username: None for username in self.influential_figures.keys()}
    
    def save_last_tweet_ids(self):
        """Save last tweet IDs to file."""
        try:
            with open('last_tweet_ids.json', 'w') as f:
                json.dump(self.last_tweet_ids, f)
        except Exception as e:
            logger.error(f"Error saving last tweet IDs: {str(e)}")
    
    def is_market_relevant(self, text: str) -> tuple[bool, List[str]]:
        """Check if tweet is market-relevant based on keywords."""
        text_lower = text.lower()
        matched_keywords = []
        
        for kw in self.market_keywords:
            if kw in text_lower:
                matched_keywords.append(kw)
        
        return len(matched_keywords) > 0, matched_keywords
    
    def fetch_new_tweets(self) -> List[Dict]:
        """Fetch new tweets from influential financial figures using snscrape."""
        tweets = []
        
        for username, name in self.influential_figures.items():
            try:
                logger.info(f"Checking for new tweets from {name} (@{username})")
                
                # Create search query
                query = f"from:{username}"
                if self.last_tweet_ids[username]:
                    # Add since_id to get only new tweets
                    query += f" since_id:{self.last_tweet_ids[username]}"
                
                # Get tweets using snscrape
                for tweet in sntwitter.TwitterSearchScraper(query).get_items():
                    # Check if tweet is market-relevant
                    is_relevant, matched_keywords = self.is_market_relevant(tweet.content)
                    if is_relevant:
                        tweets.append({
                            "source": "twitter",
                            "author": name,
                            "username": username,
                            "text": tweet.content,
                            "url": tweet.url,
                            "published": tweet.date.isoformat(),
                            "matched_keywords": matched_keywords,
                            "likes": tweet.likeCount,
                            "retweets": tweet.retweetCount,
                            "replies": tweet.replyCount
                        })
                    
                    # Update last tweet ID
                    if not self.last_tweet_ids[username] or tweet.id > int(self.last_tweet_ids[username]):
                        self.last_tweet_ids[username] = str(tweet.id)
                    
                    # Break after getting 100 tweets
                    if len(tweets) >= 100:
                        break
                
                # Add small delay to be nice to Twitter's servers
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error fetching tweets for {username}: {str(e)}")
                continue
        
        # Save updated last tweet IDs
        self.save_last_tweet_ids()
        
        return tweets
    
    def save_tweets_to_json(self, tweets: List[Dict], filename: str = "tweets.json"):
        """Save tweets to JSON file."""
        try:
            # Load existing tweets
            existing_tweets = []
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    existing_tweets = json.load(f)
            
            # Combine existing and new tweets
            all_tweets = tweets + existing_tweets
            
            # Remove duplicates based on URL
            seen_urls = set()
            unique_tweets = []
            for tweet in all_tweets:
                if tweet['url'] not in seen_urls:
                    seen_urls.add(tweet['url'])
                    unique_tweets.append(tweet)
            
            # Sort by date (newest first)
            unique_tweets.sort(key=lambda x: x['published'], reverse=True)
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump(unique_tweets, f, indent=4, default=str)
            logger.info(f"Saved {len(unique_tweets)} tweets to {filename}")
        except Exception as e:
            logger.error(f"Error saving tweets: {str(e)}")

def monitor_tweets(update_interval: int = 5):
    """Monitor tweets continuously with specified update interval in minutes."""
    collector = TweetCollector()
    
    def check_for_new_tweets():
        logger.info(f"Checking for new tweets at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        new_tweets = collector.fetch_new_tweets()
        
        if new_tweets:
            collector.save_tweets_to_json(new_tweets)
            logger.info(f"Found {len(new_tweets)} new market-relevant tweets")
            for tweet in new_tweets:
                print(f"\nNew Tweet from {tweet['author']} (@{tweet['username']})")
                print(f"Time: {tweet['published']}")
                print(f"Text: {tweet['text']}")
                print(f"Likes: {tweet['likes']} | Retweets: {tweet['retweets']} | Replies: {tweet['replies']}")
                print(f"Matched Keywords: {', '.join(tweet['matched_keywords'])}")
                print("-" * 40)
        else:
            logger.info("No new market-relevant tweets found")
    
    # Schedule the job
    schedule.every(update_interval).minutes.do(check_for_new_tweets)
    
    # Run immediately on startup
    check_for_new_tweets()
    
    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(1)

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Collect tweets from influential financial figures')
    parser.add_argument('--monitor', action='store_true', help='Run in continuous monitoring mode')
    parser.add_argument('--interval', type=int, default=5, help='Update interval in minutes (default: 5)')
    args = parser.parse_args()
    
    if args.monitor:
        logger.info(f"Starting tweet monitoring with {args.interval}-minute intervals")
        monitor_tweets(update_interval=args.interval)
    else:
        # One-time collection
        collector = TweetCollector()
        tweets = collector.fetch_new_tweets()
        collector.save_tweets_to_json(tweets)
        
        # Print summary
        logger.info(f"Successfully collected {len(tweets)} market-relevant tweets")
        for tweet in tweets[:5]:  # Show first 5 tweets as example
            print(f"\nAuthor: {tweet['author']} (@{tweet['username']})")
            print(f"Date: {tweet['published']}")
            print(f"Text: {tweet['text']}")
            print(f"Likes: {tweet['likes']} | Retweets: {tweet['retweets']} | Replies: {tweet['replies']}")
            print(f"Matched Keywords: {', '.join(tweet['matched_keywords'])}")
            print("-" * 40)

if __name__ == "__main__":
    main() 