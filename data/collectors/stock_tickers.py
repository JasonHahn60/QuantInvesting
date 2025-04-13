import requests
from typing import List

def get_all_tickers() -> List[str]:
    """
    Get complete list of all US stock tickers from GitHub repository.
    Returns a sorted list of unique tickers.
    """
    try:
        url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt"
        response = requests.get(url)
        tickers = response.text.strip().split('\n')
        print(f"Successfully retrieved {len(tickers)} tickers")
        return sorted(tickers)
    except Exception as e:
        print(f"Error getting tickers: {e}")
        return []

def save_tickers_to_file(tickers: List[str], filename: str = "all_tickers.txt"):
    """Save the list of tickers to a file."""
    try:
        with open(filename, 'w') as f:
            for ticker in tickers:
                f.write(f"{ticker}\n")
        print(f"\nSaved {len(tickers)} tickers to {filename}")
    except Exception as e:
        print(f"Error saving tickers to file: {e}")

if __name__ == "__main__":
    # Get all tickers
    all_tickers = get_all_tickers()
    
    # Save to file
    save_tickers_to_file(all_tickers)
    
    # Print first 20 tickers as example
    print("\nFirst 20 tickers:")
    print(all_tickers[:20]) 
    if "DNA" in all_tickers:
        print("DNA is in the list")
    else:
        print("DNA is not in the list")

