import praw
import pandas as pd
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import gradio as gr
import os
from dotenv import load_dotenv
import nltk

# Load environment variables
load_dotenv()

# Initialize VADER sentiment analyzer
nltk_downloader = nltk.downloader.Downloader()
if not nltk_downloader.is_installed('vader_lexicon'):
    nltk.download('vader_lexicon')

sentiment_analyzer = SentimentIntensityAnalyzer()

# Reddit API credentials
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_SECRET = os.getenv('REDDIT_SECRET')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT')

def authenticate_reddit():
    return praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )

reddit = authenticate_reddit()

def clean_post(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    return text.lower()

def scrape_reddit_posts(company, subreddit="stocks", max_posts=100):
    posts_data = []
    try:
        for submission in reddit.subreddit(subreddit).search(company, limit=max_posts):
            posts_data.append({
                'created_at': submission.created_utc,
                'title': submission.title,
                'score': submission.score,
                'num_comments': submission.num_comments,
                'text': submission.selftext
            })
    except Exception as e:
        print(f"Reddit API error: {e}")
    return pd.DataFrame(posts_data)

def preprocess_data(df):
    df = df.dropna(subset=['title'])
    df['cleaned_text'] = df['title'].apply(clean_post)
    return df.drop_duplicates(subset=['cleaned_text'])

def analyze_sentiment(df):
    df['sentiment'] = df['cleaned_text'].apply(lambda x: sentiment_analyzer.polarity_scores(x)['compound'])
    df['sentiment_label'] = df['sentiment'].apply(
        lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral')
    )
    return df

def plot_sentiment_distribution(df, company):
    sentiment_counts = df['sentiment_label'].value_counts()
    plt.figure(figsize=(6,4))
    sentiment_counts.plot(kind='bar', color=['green', 'red', 'grey'])
    plt.title(f'Sentiment Distribution for {company} (Reddit)')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Posts')
    plt.tight_layout()
    plt.savefig('sentiment_distribution.png')
    plt.close()
    return 'sentiment_distribution.png'

def stock_sentiment_analysis(company):
    df = scrape_reddit_posts(company)
    if df.empty:
        return "No posts found for the given company.", None
    df = preprocess_data(df)
    df = analyze_sentiment(df)
    if df.empty:
        return "No valid posts after preprocessing.", None
    image_path = plot_sentiment_distribution(df, company)
    positive = df[df['sentiment_label'] == 'Positive'].shape[0]
    negative = df[df['sentiment_label'] == 'Negative'].shape[0]
    neutral = df[df['sentiment_label'] == 'Neutral'].shape[0]
    return f"""
**Sentiment Analysis for {company} (Reddit):**

- **Positive Posts:** {positive}
- **Negative Posts:** {negative}
- **Neutral Posts:** {neutral}
""", image_path

iface = gr.Interface(
    fn=stock_sentiment_analysis,
    inputs=gr.Textbox(label="Enter Company Name (e.g., Apple, Tesla)", placeholder="Apple"),
    outputs=[
        gr.Markdown(label="Sentiment Summary"),
        gr.Image(label="Sentiment Distribution")
    ],
    title="Reddit Sentiment Analysis for Stocks",
    description="Enter a company name to analyze its sentiment based on recent Reddit posts."
)

if __name__ == "__main__":
    iface.launch()
