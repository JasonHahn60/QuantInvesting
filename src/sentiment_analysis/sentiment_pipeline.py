import pandas as pd
import requests
from newspaper import Article
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import feedparser
from datetime import datetime, timedelta
import numpy as np
from typing import List, Dict, Optional
import logging
import json
import os
from bs4 import BeautifulSoup
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedMarketSentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
        }
        
        # Enhanced market keywords
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
        
        # Sector-specific keywords
        self.sector_keywords = {
            'tech': ['tech', 'software', 'hardware', 'AI', 'cloud', 'semiconductor', 'chip'],
            'finance': ['bank', 'financial', 'insurance', 'investment', 'brokerage', 'wealth'],
            'energy': ['oil', 'gas', 'renewable', 'energy', 'solar', 'wind', 'fossil'],
            'healthcare': ['pharma', 'biotech', 'healthcare', 'medical', 'drug', 'vaccine'],
            'consumer': ['retail', 'consumer', 'ecommerce', 'brand', 'product', 'sales']
        }
        
    def is_market_relevant(self, title: str, text: str = "") -> tuple[bool, List[str]]:
        """Enhanced market relevance check with keyword matching."""
        title_lower = title.lower()
        text_lower = text.lower()
        matched_keywords = []
        
        # Check title first (faster)
        for kw in self.market_keywords:
            if kw in title_lower:
                matched_keywords.append(kw)
        
        # If no match in title, check text
        if not matched_keywords and text:
            for kw in self.market_keywords:
                if kw in text_lower:
                    matched_keywords.append(kw)
        
        return len(matched_keywords) > 0, matched_keywords
    
    def classify_sectors(self, text: str) -> List[str]:
        """Classify article into sectors."""
        text_lower = text.lower()
        sectors = []
        for sector, keywords in self.sector_keywords.items():
            if any(kw in text_lower for kw in keywords):
                sectors.append(sector)
        return sectors
    
    def get_article_text(self, url: str, source: str = "") -> str:
        """Enhanced article text extraction with source-specific parsing."""
        try:
            headers = self.headers.copy()
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Source-specific parsing
            if source == "investing":
                # Try multiple selectors for Investing.com
                selectors = [
                    'div.articlePage',  # Main article content
                    'div.WYSIWYG.articlePage',  # Alternative article container
                    'div.article-content'  # Another possible container
                ]
                for selector in selectors:
                    content = soup.select_one(selector)
                    if content:
                        # Remove unwanted elements
                        for element in content.select('script, style, iframe, .related-news, .comments'):
                            element.decompose()
                        return content.get_text(separator=' ', strip=True)
            
            elif source == "seeking_alpha":
                # Try multiple selectors for Seeking Alpha
                selectors = [
                    'div[data-test-id="article-content"]',  # Main article content
                    'div.article-content',  # Alternative container
                    'div.article-body'  # Another possible container
                ]
                for selector in selectors:
                    content = soup.select_one(selector)
                    if content:
                        # Remove unwanted elements
                        for element in content.select('script, style, iframe, .related-articles, .comments'):
                            element.decompose()
                        return content.get_text(separator=' ', strip=True)
            
            # Default parsing using newspaper3k
            article = Article(url)
            article.set_html(response.text)
            article.parse()
            return article.text
            
        except Exception as e:
            logger.error(f"Error extracting text from {url}: {str(e)}")
            return ""

    def get_sentiment_score(self, url: str, text: str = "", source: str = "") -> Dict:
        """Enhanced sentiment analysis with multiple scores."""
        try:
            if not text:
                text = self.get_article_text(url, source)
            
            if not text.strip():
                logger.warning(f"No text content found for article: {url}")
                return {
                    "compound": 0.0,
                    "positive": 0.0,
                    "negative": 0.0,
                    "neutral": 1.0,
                    "sectors": []
                }
            
            # Get VADER sentiment
            sentiment = self.analyzer.polarity_scores(text)
            
            # Normalize sentiment scores to ensure consistency
            total = sentiment["pos"] + sentiment["neg"] + sentiment["neu"]
            if total > 0:
                normalized_sentiment = {
                    "pos": sentiment["pos"] / total,
                    "neg": sentiment["neg"] / total,
                    "neu": sentiment["neu"] / total
                }
            else:
                normalized_sentiment = {
                    "pos": 0.0,
                    "neg": 0.0,
                    "neu": 1.0
                }
            
            # Classify sectors
            sectors = self.classify_sectors(text)
            
            # Log if sentiment is neutral (all zeros)
            if sentiment["compound"] == 0 and sentiment["pos"] == 0 and sentiment["neg"] == 0:
                logger.warning(f"Neutral sentiment detected for article: {url}")
            
            return {
                "compound": sentiment["compound"],
                "positive": normalized_sentiment["pos"],
                "negative": normalized_sentiment["neg"],
                "neutral": normalized_sentiment["neu"],
                "sectors": sectors
            }
        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            return {
                "compound": 0.0,
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
                "sectors": []
            }
    
    def fetch_reuters(self) -> List[Dict]:
        """Fetch from Reuters Business News."""
        articles = []
        try:
            url = "https://www.reuters.com/business/"
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for article in soup.find_all('article', class_='media-story-card'):
                title = article.find('h3')
                if title:
                    title_text = title.text.strip()
                    is_relevant, matched_keywords = self.is_market_relevant(title_text)
                    if is_relevant:
                        link = article.find('a')['href']
                        if not link.startswith('http'):
                            link = f"https://www.reuters.com{link}"
                        
                        articles.append({
                            "source": "reuters",
                            "title": title_text,
                            "url": link,
                            "published": datetime.now().isoformat(),
                            "summary": "",
                            "matched_keywords": matched_keywords
                        })
        except Exception as e:
            logger.error(f"Error fetching Reuters: {str(e)}")
        return articles
    
    def fetch_bloomberg(self) -> List[Dict]:
        """Fetch from Bloomberg Markets."""
        articles = []
        try:
            url = "https://www.bloomberg.com/markets"
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for article in soup.find_all('article'):
                title = article.find('h1') or article.find('h2') or article.find('h3')
                if title:
                    title_text = title.text.strip()
                    is_relevant, matched_keywords = self.is_market_relevant(title_text)
                    if is_relevant:
                        link = article.find('a')
                        if link and 'href' in link.attrs:
                            link_url = link['href']
                            if not link_url.startswith('http'):
                                link_url = f"https://www.bloomberg.com{link_url}"
                            
                            articles.append({
                                "source": "bloomberg",
                                "title": title_text,
                                "url": link_url,
                                "published": datetime.now().isoformat(),
                                "summary": "",
                                "matched_keywords": matched_keywords
                            })
        except Exception as e:
            logger.error(f"Error fetching Bloomberg: {str(e)}")
        return articles
    
    def fetch_marketwatch(self) -> List[Dict]:
        """Fetch from MarketWatch."""
        articles = []
        try:
            url = "https://www.marketwatch.com/latest-news"
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for article in soup.find_all('div', class_='article__content'):
                title = article.find('h3')
                if title:
                    title_text = title.text.strip()
                    is_relevant, matched_keywords = self.is_market_relevant(title_text)
                    if is_relevant:
                        link = article.find('a')
                        if link and 'href' in link.attrs:
                            link_url = link['href']
                            if not link_url.startswith('http'):
                                link_url = f"https://www.marketwatch.com{link_url}"
                            
                            articles.append({
                                "source": "marketwatch",
                                "title": title_text,
                                "url": link_url,
                                "published": datetime.now().isoformat(),
                                "summary": "",
                                "matched_keywords": matched_keywords
                            })
        except Exception as e:
            logger.error(f"Error fetching MarketWatch: {str(e)}")
        return articles
    
    def fetch_yahoo_rss(self) -> List[Dict]:
        """Fetch and process Yahoo Finance RSS feed."""
        articles = []
        try:
            rss_url = 'https://finance.yahoo.com/news/rssindex'
            feed = feedparser.parse(rss_url)
            
            for entry in feed.entries:
                is_relevant, matched_keywords = self.is_market_relevant(entry.title)
                if is_relevant:
                    articles.append({
                        "source": "yahoo",
                        "title": entry.title,
                        "url": entry.link,
                        "published": entry.published,
                        "summary": entry.get("summary", ""),
                        "matched_keywords": matched_keywords
                    })
        except Exception as e:
            logger.error(f"Error fetching Yahoo RSS: {str(e)}")
        return articles
    
    def fetch_investing_rss(self) -> List[Dict]:
        """Fetch and process Investing.com RSS feed."""
        articles = []
        try:
            rss_url = 'https://www.investing.com/rss/news.rss'
            response = requests.get(rss_url, headers=self.headers, timeout=10)
            feed = feedparser.parse(response.content)
            
            for entry in feed.entries:
                is_relevant, matched_keywords = self.is_market_relevant(entry.title)
                if is_relevant:
                    articles.append({
                        "source": "investing",
                        "title": entry.title,
                        "url": entry.link,
                        "published": entry.published,
                        "summary": entry.get("summary", ""),
                        "matched_keywords": matched_keywords
                    })
        except Exception as e:
            logger.error(f"Error fetching Investing RSS: {str(e)}")
        return articles
    
    def fetch_seeking_alpha(self) -> List[Dict]:
        """Fetch and process Seeking Alpha articles."""
        articles = []
        try:
            rss_url = 'https://seekingalpha.com/market_currents.xml'
            response = requests.get(rss_url, headers=self.headers, timeout=10)
            feed = feedparser.parse(response.content)
            
            for entry in feed.entries:
                is_relevant, matched_keywords = self.is_market_relevant(entry.title)
                if is_relevant:
                    articles.append({
                        "source": "seeking_alpha",
                        "title": entry.title,
                        "url": entry.link,
                        "published": entry.published,
                        "summary": entry.get("summary", ""),
                        "matched_keywords": matched_keywords
                    })
        except Exception as e:
            logger.error(f"Error fetching Seeking Alpha: {str(e)}")
        return articles
    
    def aggregate_daily_sentiment(self, articles: List[Dict]) -> Dict:
        """Enhanced daily sentiment aggregation with standardized metrics."""
        if not articles:
            return {
                "sentiment_avg": 0.0,
                "sentiment_std": 0.0,
                "sentiment_min": 0.0,
                "sentiment_max": 0.0,
                "article_count": 0,
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
                "neutral_ratio": 0.0,
                "sector_distribution": {},
                "source_distribution": {},
                "non_zero_sentiment_avg": 0.0,
                "zero_sentiment_count": 0
            }
        
        # Extract sentiment scores
        compound_scores = [article["sentiment"]["compound"] for article in articles]
        positive_scores = [article["sentiment"]["positive"] for article in articles]
        negative_scores = [article["sentiment"]["negative"] for article in articles]
        neutral_scores = [article["sentiment"]["neutral"] for article in articles]
        
        # Calculate statistics for compound scores (already normalized between -1 and 1)
        sentiment_avg = np.mean(compound_scores)
        sentiment_std = np.std(compound_scores)
        sentiment_min = min(compound_scores)
        sentiment_max = max(compound_scores)
        
        # Calculate non-zero sentiment average
        non_zero_scores = [score for score in compound_scores if score != 0]
        non_zero_sentiment_avg = np.mean(non_zero_scores) if non_zero_scores else 0.0
        zero_sentiment_count = len(compound_scores) - len(non_zero_scores)
        
        # Calculate ratios using normalized scores
        positive_ratio = np.mean(positive_scores)
        negative_ratio = np.mean(negative_scores)
        neutral_ratio = np.mean(neutral_scores)
        
        # Calculate sector distribution
        sector_counts = {}
        for article in articles:
            for sector in article["sentiment"].get("sectors", []):
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        # Calculate source distribution
        source_counts = {}
        for article in articles:
            source = article["source"]
            source_counts[source] = source_counts.get(source, 0) + 1
        
        return {
            "sentiment_avg": sentiment_avg,
            "sentiment_std": sentiment_std,
            "sentiment_min": sentiment_min,
            "sentiment_max": sentiment_max,
            "article_count": len(compound_scores),
            "positive_ratio": positive_ratio,
            "negative_ratio": negative_ratio,
            "neutral_ratio": neutral_ratio,
            "sector_distribution": sector_counts,
            "source_distribution": source_counts,
            "non_zero_sentiment_avg": non_zero_sentiment_avg,
            "zero_sentiment_count": zero_sentiment_count
        }
    
    def process_articles(self, articles: List[Dict]) -> pd.DataFrame:
        """Process articles with enhanced sentiment analysis."""
        processed_articles = []
        for article in articles:
            # Get sentiment and sector classification
            sentiment = self.get_sentiment_score(
                article["url"], 
                text=article.get("summary", ""), 
                source=article["source"]
            )
            article["sentiment"] = sentiment
            processed_articles.append(article)
            # Add delay to avoid rate limiting
            time.sleep(1)
        
        df = pd.DataFrame(processed_articles)
        if not df.empty:
            df["published"] = pd.to_datetime(df["published"], format='mixed', utc=True)
            df = df.sort_values("published")
        
        return df
    
    def get_daily_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get enhanced daily sentiment features."""
        if df.empty:
            return pd.DataFrame()
        
        # Group by date and apply aggregation
        daily_features = df.groupby(df["published"].dt.date).apply(
            lambda x: pd.Series(self.aggregate_daily_sentiment(x.to_dict("records")))
        ).reset_index()
        
        return daily_features

    def save_articles_to_json(self, articles: List[Dict], filename: str = "articles.json"):
        """Save articles to JSON file."""
        with open(filename, 'w') as f:
            json.dump(articles, f, indent=4, default=str)

    def load_articles_from_json(self, filename: str = "articles.json") -> List[Dict]:
        """Load articles from JSON file."""
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return json.load(f)
        return []

    def calculate_sentiment_momentum(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """Calculate sentiment momentum indicators."""
        # Ensure data is sorted by time
        df = df.sort_values('published')
        
        # Calculate rolling sentiment metrics
        df['sentiment_ma'] = df['sentiment'].rolling(window=window).mean()
        df['sentiment_std'] = df['sentiment'].rolling(window=window).std()
        df['sentiment_momentum'] = df['sentiment'].diff(window)
        df['sentiment_velocity'] = df['sentiment_momentum'] / window
        
        # Calculate sentiment z-score
        df['sentiment_zscore'] = (df['sentiment'] - df['sentiment_ma']) / df['sentiment_std']
        
        return df

    def detect_sentiment_divergence(self, df: pd.DataFrame, price_data: pd.DataFrame) -> pd.DataFrame:
        """Detect divergences between sentiment and price."""
        # Merge sentiment and price data
        merged = pd.merge_asof(
            df.sort_values('published'),
            price_data.sort_values('date'),
            left_on='published',
            right_on='date',
            direction='nearest'
        )
        
        # Calculate price momentum
        merged['price_momentum'] = merged['close'].pct_change(5)
        
        # Detect divergences
        merged['sentiment_price_divergence'] = (
            (merged['sentiment_momentum'] > 0) & (merged['price_momentum'] < 0) |
            (merged['sentiment_momentum'] < 0) & (merged['price_momentum'] > 0)
        )
        
        return merged

    def analyze_sector_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze sentiment by sector for rotation strategies."""
        # Explode sectors into separate rows
        sector_df = df.explode('sectors')
        
        # Calculate sector sentiment metrics
        sector_metrics = sector_df.groupby('sectors').agg({
            'sentiment': ['mean', 'std', 'count'],
            'sentiment_momentum': 'mean',
            'sentiment_velocity': 'mean'
        }).round(3)
        
        # Calculate sector rankings
        sector_metrics['sentiment_rank'] = sector_metrics[('sentiment', 'mean')].rank(ascending=False)
        sector_metrics['momentum_rank'] = sector_metrics[('sentiment_momentum', 'mean')].rank(ascending=False)
        
        return sector_metrics

    def generate_trading_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on sentiment analysis."""
        signals = pd.DataFrame(index=df.index)
        
        # Momentum signals
        signals['momentum_signal'] = np.where(
            (df['sentiment_momentum'] > 0) & (df['sentiment_zscore'] > 1),
            1,  # Strong positive momentum
            np.where(
                (df['sentiment_momentum'] < 0) & (df['sentiment_zscore'] < -1),
                -1,  # Strong negative momentum
                0  # Neutral
            )
        )
        
        # Divergence signals
        signals['divergence_signal'] = np.where(
            df['sentiment_price_divergence'],
            np.where(df['sentiment_momentum'] > 0, 1, -1),
            0
        )
        
        # Combined signal
        signals['combined_signal'] = signals['momentum_signal'] + signals['divergence_signal']
        
        return signals

# Example usage
if __name__ == "__main__":
    analyzer = EnhancedMarketSentimentAnalyzer()
    
    # Fetch and process articles
    all_articles = []
    all_articles.extend(analyzer.fetch_reuters())
    all_articles.extend(analyzer.fetch_bloomberg())
    all_articles.extend(analyzer.fetch_marketwatch())
    all_articles.extend(analyzer.fetch_yahoo_rss())
    all_articles.extend(analyzer.fetch_investing_rss())
    all_articles.extend(analyzer.fetch_seeking_alpha())
    
    # Process articles
    df = analyzer.process_articles(all_articles)
    
    # Calculate sentiment momentum
    df = analyzer.calculate_sentiment_momentum(df)
    
    # Generate trading signals
    signals = analyzer.generate_trading_signals(df)
    
    # Analyze sector sentiment
    sector_metrics = analyzer.analyze_sector_sentiment(df)
    
    print("\n=== TRADING SIGNALS ===")
    print("-" * 80)
    print("\nRecent Trading Signals:")
    recent_signals = signals.tail(5)
    for idx, row in recent_signals.iterrows():
        print(f"\nDate: {idx}")
        print(f"Momentum Signal: {row['momentum_signal']}")
        print(f"Divergence Signal: {row['divergence_signal']}")
        print(f"Combined Signal: {row['combined_signal']}")
    
    print("\n=== SECTOR ANALYSIS ===")
    print("-" * 80)
    print("\nSector Sentiment Rankings:")
    for sector, metrics in sector_metrics.iterrows():
        print(f"\nSector: {sector}")
        print(f"Average Sentiment: {metrics[('sentiment', 'mean')]:.3f}")
        print(f"Sentiment Momentum: {metrics[('sentiment_momentum', 'mean')]:.3f}")
        print(f"Article Count: {metrics[('sentiment', 'count')]}")
        print(f"Sentiment Rank: {metrics['sentiment_rank']:.0f}")
        print(f"Momentum Rank: {metrics['momentum_rank']:.0f}")
    
    print("\n=== DETAILED ARTICLE LIST ===")
    print("-" * 80)
    for article in all_articles:
        print(f"\nSource: {article['source']}")
        print(f"Title: {article['title']}")
        print(f"URL: {article['url']}")
        print(f"Matched Keywords: {', '.join(article['matched_keywords'])}")
        sentiment = article['sentiment']
        print(f"Sentiment: {sentiment['compound']:.3f}")
        print(f"Positive: {sentiment['positive']:.3f}")
        print(f"Negative: {sentiment['negative']:.3f}")
        print(f"Neutral: {sentiment['neutral']:.3f}")
        if sentiment['sectors']:
            print(f"Sectors: {', '.join(sentiment['sectors'])}")
        print("-" * 40) 