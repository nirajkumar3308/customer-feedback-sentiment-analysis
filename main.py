import csv
import string
from datetime import datetime
import pandas as pd
import numpy as np
import unittest

# =========================
# SIMPLE SENTIMENT LOGIC
# =========================
positive_words = ["good", "great", "amazing", "love", "excellent", "best"]
negative_words = ["bad", "worst", "hate", "waste", "poor", "terrible"]

def simple_sentiment(text):
    text = text.lower()
    pos = sum(word in text for word in positive_words)
    neg = sum(word in text for word in negative_words)

    if pos > neg:
        return "Positive"
    elif neg > pos:
        return "Negative"
    else:
        return "Neutral"

# =========================
# REVIEW CLASS
# =========================
class Review:
    def __init__(self, cid, pid, rating, comment, date=None):
        if not comment.strip():
            raise ValueError("Review cannot be empty")

        self.customer_id = cid
        self.product_id = pid
        self.rating = rating
        self.comment = comment
        self.date = date if date else datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def save_to_file(self, filename="reviews.csv"):
        with open(filename, "a", newline="") as file:
            writer = csv.writer(file)

            if file.tell() == 0:
                writer.writerow(["Customer ID", "Product ID", "Review", "Date", "Rating"])

            writer.writerow([self.customer_id, self.product_id, self.comment, self.date, self.rating])

# =========================
# PRODUCT CLASS
# =========================
class Product:
    def __init__(self, pid, name, price):
        self.product_id = pid
        self.name = name
        self.price = price
        self.reviews = []

    def add_review(self, review):
        self.reviews.append(review)

# =========================
# DECORATOR
# =========================
def preprocess_review(func):
    def wrapper(self, review):
        review.comment = review.comment.lower().translate(
            str.maketrans('', '', string.punctuation)
        )
        return func(self, review)
    return wrapper

# =========================
# SENTIMENT ANALYZER
# =========================
class SimpleSentimentAnalyzer:
    @preprocess_review
    def analyze_sentiment(self, review):
        return simple_sentiment(review.comment)

# =========================
# SAMPLE PREDICTION
# =========================
def run_sample_prediction():
    analyzer = SimpleSentimentAnalyzer()

    sample_reviews = [
        "This product is amazing",
        "This is worst product",
        "It is okay"
    ]

    for text in sample_reviews:
        review = Review(1, 101, 5, text)
        review.save_to_file()

        sentiment = analyzer.analyze_sentiment(review)
        print("Review:", text)
        print("Sentiment:", sentiment)
        print()

# =========================
# PANDAS ANALYSIS
# =========================
def load_reviews():
    return pd.read_csv("reviews.csv")

def classify_sentiment(review):
    result = simple_sentiment(review)
    return 1 if result == "Positive" else -1 if result == "Negative" else 0

def generate_insights(df):
    df["Sentiment Score"] = df["Review"].apply(classify_sentiment)

    overall = df["Sentiment Score"].mean()
    ratings = df.groupby("Product ID")["Rating"].mean()

    most_positive = df.groupby("Product ID")["Sentiment Score"].mean().idxmax()
    most_negative = df.groupby("Product ID")["Sentiment Score"].mean().idxmin()

    return overall, ratings, most_positive, most_negative

# =========================
# NUMPY STATS
# =========================
def numpy_stats(df):
    ratings = df["Rating"].values
    return np.mean(ratings), np.median(ratings), pd.Series(ratings).mode()[0]

# =========================
# UNIT TESTS
# =========================
class TestSystem(unittest.TestCase):

    def test_review(self):
        r = Review(1, 101, 5, "Good product")
        self.assertEqual(r.rating, 5)

    def test_empty_review(self):
        with self.assertRaises(ValueError):
            Review(1, 101, 5, "")

    def test_sentiment(self):
        r = Review(1, 101, 5, "Amazing product")
        analyzer = SimpleSentimentAnalyzer()
        self.assertEqual(analyzer.analyze_sentiment(r), "Positive")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("===== CUSTOMER FEEDBACK SYSTEM =====\n")

    run_sample_prediction()

    try:
        df = load_reviews()
        overall, ratings, pos, neg = generate_insights(df)

        print("--- ANALYSIS ---")
        print("Overall Satisfaction:", overall)
        print("Ratings:\n", ratings)
        print("Most Positive Product:", pos)
        print("Most Negative Product:", neg)

        mean, median, mode = numpy_stats(df)
        print("Mean:", mean, "Median:", median, "Mode:", mode)

    except:
        print("No data available.")

    print("\nRunning Tests...\n")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
