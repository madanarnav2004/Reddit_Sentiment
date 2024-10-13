# Reddit Sentiment Analysis for Stocks

This project provides an interface to analyze Reddit posts about specific companies using sentiment analysis. It scrapes posts from the r/stocks subreddit, processes them using NLTK’s VADER sentiment analyzer, and visualizes the sentiment distribution in a simple Gradio interface.

## Table of Contents

- [Features](#features)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Dependencies](#dependencies)

## Features

- **Reddit API Integration**: Fetches posts from Reddit's r/stocks subreddit.
- **Sentiment Analysis**: Uses NLTK's VADER for polarity and sentiment scoring.
- **Data Visualization**: Displays a bar chart with sentiment distribution.
- **Interactive Interface**: Gradio-based UI for easy interaction.

---

## Setup Instructions

### Prerequisites

Make sure you have **Python 3.7** or higher installed on your system.

### 1. Clone the Repository

Clone this repository to your local machine using the following command:

```bash
git clone https://github.com/madanarnav2004/Reddit_Sentiment.git
cd reddit-sentiment-analysis
```
## Usage

### Running the Application

To start the Gradio interface for sentiment analysis, run the following:

```bash
python main.py
```

### How It Works

1. **Authenticate with Reddit**: Uses the PRAW library to authenticate with Reddit's API.
2. **Scrape Posts**: Queries the subreddit `r/stocks` based on the company name provided.
3. **Clean Post Titles**: Removes URLs, special characters, and converts the text to lowercase for consistency.
4. **Sentiment Analysis**: NLTK’s VADER model is used to score each post's sentiment as Positive, Negative, or Neutral.
5. **Visualize Sentiment**: Generates a bar chart showing sentiment distribution and provides a summary.


### Dependencies

- **[PRAW](https://praw.readthedocs.io/)**: Python Reddit API Wrapper
- **[Pandas](https://pandas.pydata.org/)**: Data manipulation and analysis
- **[NLTK](https://www.nltk.org/)**: Sentiment analysis with the VADER lexicon
- **[Matplotlib](https://matplotlib.org/)**: Visualization of sentiment distribution
- **[Gradio](https://gradio.app/)**: Web interface for interactive machine learning models
- **[python-dotenv](https://pypi.org/project/python-dotenv/)**: Loads environment variables from a `.env` file

To install all dependencies, simply run:

```bash
pip install -r requirements.txt
```
