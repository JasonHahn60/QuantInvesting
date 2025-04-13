import pandas as pd
import tweepy
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
import json
import os
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TwitterSentimentAnalyzer:
    def __init__(self, 
                 twitter_api_key: str,
                 twitter_api_secret: str,
                 twitter_access_token: str,
                 twitter_access_token_secret: str):
        """Initialize Twitter sentiment analyzer with API credentials."""
        self.analyzer = SentimentIntensityAnalyzer()
        
        # Initialize Twitter API
        auth = tweepy.OAuthHandler(twitter_api_key, twitter_api_secret)
        auth.set_access_token(twitter_access_token, twitter_access_token_secret)
        self.twitter_client = tweepy.API(auth, wait_on_rate_limit=True)
        
        # List of influential financial figures to track
        self.influential_figures = {
            'elonmusk': 'Elon Musk',
            'realDonaldTrump': 'Donald Trump',
            'cathiewood': 'Cathie Wood',
            'chamath': 'Chamath Palihapitiya',
            'raydalio': 'Ray Dalio',
            'michaeljburry': 'Michael Burry',
            'charliebilello': 'Charlie Bilello',
            'morganhousel': 'Morgan Housel',
            'howardmarks': 'Howard Marks',
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
            "bull", "bear", "rally", "correction", "crash", "volatility"
        ]
    
    def is_market_relevant(self, text: str) -> tuple[bool, List[str]]:
        """Check if tweet is market-relevant based on keywords."""
        text_lower = text.lower()
        matched_keywords = []
        
        for kw in self.market_keywords:
            if kw in text_lower:
                matched_keywords.append(kw)
        
        return len(matched_keywords) > 0, matched_keywords
    
    def analyze_tweet_sentiment(self, tweet: str) -> Dict:
        """Enhanced sentiment analysis for tweets using both VADER and TextBlob."""
        # VADER sentiment
        vader_sentiment = self.analyzer.polarity_scores(tweet)
        
        # TextBlob sentiment
        blob = TextBlob(tweet)
        textblob_sentiment = {
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity
        }
        
        # Combine both analyses
        combined_sentiment = {
            "compound": (vader_sentiment["compound"] + textblob_sentiment["polarity"]) / 2,
            "positive": (vader_sentiment["pos"] + (textblob_sentiment["polarity"] + 1) / 2) / 2,
            "negative": (vader_sentiment["neg"] + (-textblob_sentiment["polarity"] + 1) / 2) / 2,
            "neutral": vader_sentiment["neu"],
            "subjectivity": textblob_sentiment["subjectivity"]
        }
        
        return combined_sentiment
    
    def fetch_influencer_tweets(self, days_back: int = 7) -> List[Dict]:
        """Fetch recent tweets from influential financial figures."""
        tweets = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        for username, name in self.influential_figures.items():
            try:
                # Fetch user's recent tweets
                user_tweets = self.twitter_client.user_timeline(
                    screen_name=username,
                    count=200,  # Maximum allowed per request
                    tweet_mode='extended',
                    exclude_replies=True,
                    include_rts=False
                )
                
                for tweet in user_tweets:
                    tweet_date = tweet.created_at
                    if start_date <= tweet_date <= end_date:
                        # Check if tweet is market-relevant
                        is_relevant, matched_keywords = self.is_market_relevant(tweet.full_text)
                        if is_relevant:
                            sentiment = self.analyze_tweet_sentiment(tweet.full_text)
                            tweets.append({
                                "source": "twitter",
                                "author": name,
                                "username": username,
                                "text": tweet.full_text,
                                "url": f"https://twitter.com/{username}/status/{tweet.id}",
                                "published": tweet_date.isoformat(),
                                "sentiment": sentiment,
                                "matched_keywords": matched_keywords
                            })
                            
            except Exception as e:
                logger.error(f"Error fetching tweets for {username}: {str(e)}")
                continue
                
        return tweets
    
    def process_tweets(self, tweets: List[Dict]) -> pd.DataFrame:
        """Process tweets into a DataFrame with sentiment metrics."""
        if not tweets:
            return pd.DataFrame()
            
        df = pd.DataFrame(tweets)
        df['published'] = pd.to_datetime(df['published'])
        df = df.sort_values('published')
        
        # Extract sentiment metrics
        df['compound_sentiment'] = df['sentiment'].apply(lambda x: x['compound'])
        df['subjectivity'] = df['sentiment'].apply(lambda x: x['subjectivity'])
        
        return df
    
    def calculate_tweet_momentum(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """Calculate sentiment momentum indicators for tweets."""
        if df.empty:
            return df
            
        # Group by author and calculate rolling metrics
        df['sentiment_ma'] = df.groupby('author')['compound_sentiment'].rolling(
            window=window, min_periods=1
        ).mean().reset_index(0, drop=True)
        
        df['sentiment_std'] = df.groupby('author')['compound_sentiment'].rolling(
            window=window, min_periods=1
        ).std().reset_index(0, drop=True)
        
        df['sentiment_momentum'] = df.groupby('author')['compound_sentiment'].diff(window)
        
        return df
    
    def save_tweets_to_json(self, tweets: List[Dict], filename: str = "tweets.json"):
        """Save tweets to JSON file."""
        with open(filename, 'w') as f:
            json.dump(tweets, f, indent=4, default=str)
    
    def load_tweets_from_json(self, filename: str = "tweets.json") -> List[Dict]:
        """Load tweets from JSON file."""
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return json.load(f)
        return []

# Example usage
if __name__ == "__main__":
    # Initialize analyzer with Twitter credentials
    analyzer = TwitterSentimentAnalyzer(
        twitter_api_key="YOUR_API_KEY",
        twitter_api_secret="YOUR_API_SECRET",
        twitter_access_token="YOUR_ACCESS_TOKEN",
        twitter_access_token_secret="YOUR_ACCESS_TOKEN_SECRET"
    )
    
    # Fetch recent tweets
    tweets = analyzer.fetch_influencer_tweets(days_back=7)
    
    # Process tweets
    df = analyzer.process_tweets(tweets)
    
    # Calculate momentum
    df = analyzer.calculate_tweet_momentum(df)
    
    # Save tweets
    analyzer.save_tweets_to_json(tweets)
    
    # Print results
    print("\n=== RECENT INFLUENCER TWEETS ===")
    print("-" * 80)
    for _, tweet in df.tail(5).iterrows():  # Show 5 most recent tweets
        print(f"\nAuthor: {tweet['author']} (@{tweet['username']})")
        print(f"Date: {tweet['published']}")
        print(f"Text: {tweet['text']}")
        print(f"Sentiment: {tweet['compound_sentiment']:.3f}")
        print(f"Subjectivity: {tweet['subjectivity']:.3f}")
        print(f"Sentiment Momentum: {tweet['sentiment_momentum']:.3f}")
        print(f"Matched Keywords: {', '.join(tweet['matched_keywords'])}")
        print("-" * 40) 