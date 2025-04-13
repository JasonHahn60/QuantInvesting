"""
Sentiment analysis tools and utilities.
"""
import logging
from typing import Dict, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Analyzes and processes sentiment data."""
    
    def __init__(self):
        """Initialize the sentiment analyzer."""
        pass
    
    def aggregate_sentiment(self, sentiment_data: List[Dict], window: str = '1D') -> Dict:
        """Aggregate sentiment data over a time window."""
        try:
            df = pd.DataFrame(sentiment_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Resample and calculate mean sentiment
            resampled = df.resample(window).agg({
                'polarity': 'mean',
                'subjectivity': 'mean'
            })
            
            return {
                'mean_polarity': resampled['polarity'].mean(),
                'mean_subjectivity': resampled['subjectivity'].mean(),
                'count': len(sentiment_data)
            }
        except Exception as e:
            logger.error(f"Error aggregating sentiment: {str(e)}")
            return {'mean_polarity': 0, 'mean_subjectivity': 0, 'count': 0}
    
    def detect_sentiment_shifts(self, sentiment_data: List[Dict], threshold: float = 0.5) -> List[Dict]:
        """Detect significant shifts in sentiment."""
        shifts = []
        for i in range(1, len(sentiment_data)):
            prev = sentiment_data[i-1]
            curr = sentiment_data[i]
            
            polarity_change = abs(curr['polarity'] - prev['polarity'])
            if polarity_change > threshold:
                shifts.append({
                    'timestamp': curr['timestamp'],
                    'polarity_change': polarity_change,
                    'previous_polarity': prev['polarity'],
                    'current_polarity': curr['polarity']
                })
        
        return shifts
    
    def correlate_with_market(self, sentiment_data: List[Dict], price_data: pd.DataFrame) -> Dict:
        """Correlate sentiment with market price movements."""
        try:
            # Convert sentiment data to DataFrame
            sentiment_df = pd.DataFrame(sentiment_data)
            sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'])
            
            # Merge with price data
            merged = pd.merge_asof(
                sentiment_df.sort_values('timestamp'),
                price_data.sort_values('timestamp'),
                on='timestamp'
            )
            
            # Calculate correlations
            correlations = {
                'polarity_price': merged['polarity'].corr(merged['price']),
                'subjectivity_price': merged['subjectivity'].corr(merged['price'])
            }
            
            return correlations
        except Exception as e:
            logger.error(f"Error correlating with market: {str(e)}")
            return {'polarity_price': 0, 'subjectivity_price': 0} 