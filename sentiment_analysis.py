from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

# Load your dataset
df = pd.read_csv('wine_final.csv')

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Define a function to extract the compound sentiment score, converting non-strings to empty strings
def get_sentiment_score(description):
    if not isinstance(description, str):  # If it's not a string, convert it to an empty string
        description = ""
    return analyzer.polarity_scores(description)['compound']

# Apply the function to get the sentiment score
df['Sentiment_Score'] = df['description'].apply(get_sentiment_score)

# Save the result
df.to_csv('wine_updated.csv', index=False)
print("finished")