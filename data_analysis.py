import os
import certifi
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from jina_embedding import index  # Ensure this is updated for GPT-2

# Set SSL certificate environment variable
os.environ['SSL_CERT_FILE'] = certifi.where()

# Load the CSV file
file_path = '/Users/kristophermolter/PycharmProjects/pythonProject/venv/date_filtered_SBI_STTR_award_data.csv'
df = pd.read_csv(file_path)

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Set up stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to clean text
def clean_text(text):
    if not isinstance(text, str):
        return ""  # Return empty string for non-string values
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Apply text cleaning to Abstract and Award Title
df['Abstract_Cleaned'] = df['Abstract'].apply(clean_text)
df['Award_Title_Cleaned'] = df['Award Title'].apply(clean_text)

# List of keywords to search for
keywords = [
    "Detect and Avoid Systems", "Electric Power Trains",
    "Certification of autonomous software", "Ground Clutter",
    "RADAR", "Ground", "Auto", "Air-Air Collision Avoidance"
]

# Function to check if any keyword is in the text
def contains_keywords(text, keywords):
    return any(keyword.lower() in text for keyword in keywords)

# Searching for keywords in the cleaned 'Abstract' and 'Award Title' fields
df['Abstract_Contains_Keywords'] = df['Abstract_Cleaned'].apply(lambda x: contains_keywords(x, keywords))
df['Award_Title_Contains_Keywords'] = df['Award_Title_Cleaned'].apply(lambda x: contains_keywords(x, keywords))

# Filter and display specific columns
columns_to_display = ['Company', 'Award Title', 'Abstract', 'Abstract_Contains_Keywords', 'Award_Title_Contains_Keywords']
display_df = df[df['Abstract_Contains_Keywords'] | df['Award_Title_Contains_Keywords']][columns_to_display]
print(display_df)

# Embedding the cleaned text using GPT-2 and storing the embeddings
df['Abstract_Embeddings'] = df['Abstract_Cleaned'].apply(index)

# Visualization: Keyword Frequency
keyword_counts = {keyword: df['Abstract_Cleaned'].str.contains(keyword, case=False).sum() for keyword in keywords}
plt.figure(figsize=(10, 6))
plt.bar(keyword_counts.keys(), keyword_counts.values())
plt.xlabel('Keywords')
plt.ylabel('Frequency')
plt.title('Keyword Frequency in Abstracts')
plt.xticks(rotation=45)
plt.show()

# Visualization: Word Cloud for Abstracts
wordcloud = WordCloud(width=800, height=800, background_color='white', stopwords=stop_words, min_font_size=10).generate(' '.join(df['Abstract_Cleaned']))
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

# Optional: Save the DataFrame with embeddings to a new CSV file
df.to_csv('embedded_data.csv', index=False)
