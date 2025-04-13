import os
import shutil

def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def move_file(src, dest):
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(src, dest)
        print(f"Moved {src} to {dest}")
    else:
        print(f"Source file {src} does not exist")

def organize_files():
    # Create necessary directories
    directories = [
        "src/data_collection",
        "src/sentiment_analysis",
        "src/database",
        "src/utils",
        "data"
    ]
    
    for directory in directories:
        create_directory_if_not_exists(directory)

    # Move files to appropriate locations
    file_moves = [
        # Move data collection related files
        ("Data_Collection.py", "src/data_collection/data_collection.py"),
        ("collect_tweets.py", "src/data_collection/collect_tweets.py"),
        ("stock_tickers.py", "src/data_collection/stock_tickers.py"),
        
        # Move sentiment analysis related files
        ("sentiment_pipeline.py", "src/sentiment_analysis/sentiment_pipeline.py"),
        ("twitter_sentiment.py", "src/sentiment_analysis/twitter_sentiment.py"),
        
        # Move database related files
        ("stock_database.py", "src/database/stock_database.py"),
        
        # Move data files
        ("tweets.json", "data/tweets.json"),
        ("articles.json", "data/articles.json"),
        ("last_tweet_ids.json", "data/last_tweet_ids.json"),
        ("all_tickers.txt", "data/all_tickers.txt"),
        ("stock_data.db", "data/stock_data.db"),
    ]

    for src, dest in file_moves:
        move_file(src, dest)

if __name__ == "__main__":
    organize_files() 