import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats, signal

def classify_trend_with_quality(close_prices, window=1250):
    """Classify stock trend into 5 buckets using 5 years of recent data"""
    if len(close_prices) < 5000: # need 20 years of data in training
        return None
    
    recent = close_prices[-window:].dropna().values.ravel()
    if len(recent) < window * 0.8:  # Need 80% valid data
        return None
    
    # Linear trend on log prices (compound annual growth)
    log_prices = np.log(recent)
    slope, _, _, _, _ = stats.linregress(np.arange(len(log_prices)), log_prices)
    daily_returns = np.diff(recent) / recent[:-1]
    vol = np.std(daily_returns)
    
    # Spike detection (prominent peaks)
    peaks, _ = signal.find_peaks(recent, prominence=np.std(recent) * 1.5)
    n_peaks = len(peaks)

    # Establish quality score to rank classification results
    # Will be used to pick 10 best candidates per bucket
    quality = 0.0
    
    # Classification rules (thresholds tuned for clear separation)
    if slope > 0.0004 and vol < 0.03:
        quality = (slope - 0.0004) * 1000 - (vol - 0.02) * 50  # Reward growth, punish vol
        return ('increasing', max(quality, 0))
    elif slope < -0.0004 and vol < 0.03:
        quality = (-slope - 0.0004) * 1000 - (vol - 0.02) * 50  # Reward decline magnitude
        return ('decreasing', max(quality, 0))
    elif n_peaks == 1:
        quality = 1.0 / (vol + 0.01)  # Reward single clean spike
        return ('one_spike', quality)
    elif n_peaks == 2:
        quality = 1.0 / (vol + 0.01)  # Reward double spikes
        return ('two_spikes', quality)
    elif abs(slope) < 0.0003 and vol < 0.02:
        quality = (0.02 - vol) * 100  # Reward lower volatility
        return ('stagnant', quality)
    return None

# 1. Load OFFICIAL NASDAQ tickers (~5200 real symbols)
print("Loading real NASDAQ tickers...")
nasdaq_df = pd.read_csv('https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt', sep='|')
tickers = [sym for sym in nasdaq_df['Symbol'].tolist() if len(str(sym)) <= 5]  # Filter valid tickers
print(f"Loaded {len(tickers)} real NASDAQ tickers")

# 2. Initialize buckets with early stopping
quality_buckets = {cat: [] for cat in ['increasing', 'decreasing', 'one_spike', 'two_spikes', 'stagnant']}
MAX_PER_BUCKET = 15

# 3. Index-based sampling with early cutoffs
print("\nSampling stocks and classifying trends...")
for i, ticker in enumerate(tickers):
    
    try:
        # Fetch 20 years data (~5000 trading days)
        stock = yf.download(ticker, start='2005-09-01', end='2025-09-01', progress=False, auto_adjust=True)['Close']
        
        result = classify_trend_with_quality(stock)
        
        if result:
            category, quality = result
            quality_buckets[category].append((ticker, quality))
            quality_buckets[category] = sorted(quality_buckets[category], key=lambda x: x[1], reverse=True)[:MAX_PER_BUCKET]
            
    except Exception as e:
        # Silently skip invalid/delisted tickers
        print(f"✗ {ticker}: Error fetching data ({e})")
        break

# 4. Save results
results_df = pd.DataFrame([
    {'category': cat, 'ticker': ticker, 'quality': quality}
    for cat, tickers_list in quality_buckets.items()
    for ticker, quality in tickers_list
])
results_df.to_csv('selected_stocks_quality.csv', index=True)
print("\n" + "="*60)
print("FINAL RESULTS:")
print(results_df.to_string(index=True))
print("="*60)
print("Saved to 'selected_stocks_quality.csv'")
print("="*60)
print("Downloading selected stocks data...")

df = pd.read_csv('selected_stocks_quality.csv')
tickers = df['ticker'].unique().tolist()
data = yf.download(tickers, start='2005-09-01', end='2025-09-01', progress=True, auto_adjust=True)['Close']
data.to_csv('selected_stocks_data.csv')

print("Saved to 'selected_stocks_data.csv'")
print("="*60)