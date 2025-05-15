from nltk import pos_tag
from nltk.corpus import wordnet
import re, nltk
# nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt'); nltk.download('punkt_tab');nltk.download('averaged_perceptron_tagger_eng')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import kagglehub
import os
import pandas as pd
import contractions
from pandasgui import show
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
import joblib
from imblearn.over_sampling import RandomOverSampler
import numpy as np
# Stopwords are common words that do not add much meaning to the text
# and are often removed in text preprocessing.
stop_words = set(stopwords.words('english'))
# lemmatization is the process of reducing a word to its base or root form.
# For example, "running" becomes "run", "better" becomes "good".
lemmatizer = WordNetLemmatizer()

# Function to convert NLTK POS tag to WordNet POS tag
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN 

def cleaning(text):
    text = contractions.fix(text)
    text = re.sub(r'@\w+|http\S+', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower().strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    tagged_tokens = pos_tag(tokens)
    lemmatized = [
        lemmatizer.lemmatize(w, get_wordnet_pos(pos)) 
        for w, pos in tagged_tokens
    ]
    return " ".join(lemmatized)

# Function to evaluate and visualize each model
# Confusion matrix and feature importance
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    print(f"\n=== {name} ===")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("\nClassification Report:\n")
    print(classification_report(y_test, preds, target_names=["negative", "neutral", "positive"]))

    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['neg', 'neu', 'pos'], yticklabels=['neg', 'neu', 'pos'])
    plt.title(f"Confusion Matrix: {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    if hasattr(model, 'coef_'):
        feature_names = np.array(tfidf.get_feature_names_out())
        for i, class_label in enumerate(['negative', 'neutral', 'positive']):
            top_pos_idx = np.argsort(model.coef_[i])[-10:][::-1]
            print(f"\nTop words for {class_label}:")
            print(feature_names[top_pos_idx])

def predict_sentiment(tweet):
    cleaned = cleaning(tweet)
    vector = tfidf.transform([cleaned])
    prediction = best_model.predict(vector)[0]
    return ['negative', 'neutral', 'positive'][prediction]

def eda(df):
     # Exploratory Data Analysis (EDA)
    print("Total tweets:", len(df))
    print("Class distribution:")
    print(df['airline_sentiment'].value_counts())

    df['tweet_length'] = df['text'].apply(lambda x: len(x.split()))
    print("\nAverage tweet length:", df['tweet_length'].mean())

    print("\nExample tweets per sentiment:")
    for label in ['negative', 'neutral', 'positive']:
        example = df[df['airline_sentiment'] == label]['text'].iloc[0]
        print(f"{label.capitalize()}: {example}\n")

    sns.countplot(data=df, x='airline_sentiment', order=['negative', 'neutral', 'positive'])
    plt.title("Sentiment Class Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Tweet Count")
    plt.show()

if __name__ == "__main__":
    # Download the dataset from Kaggle
    path = kagglehub.dataset_download("crowdflower/twitter-airline-sentiment")
    # Load the dataset
    file = os.path.join(path, "Tweets.csv")
    df = pd.read_csv(file)
    # Drop unnecessary columns and rows with missing values
    df = df[["text", "airline_sentiment"]].dropna()

    eda(df)

    # Data cleaning
    df["cleaning"] = df["text"].apply(cleaning)
    df['label'] = df['airline_sentiment'].map({'negative':0, 'neutral':1, 'positive':2})
    show(df.head())
    X = df['cleaning']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42)
    # print("Class proportions:\n", y_train.value_counts(normalize=True))  # verify class proportions

    # Vectorization
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
    X_train_tfidf = tfidf.fit_transform(X_train)    # learn vocabulary from training data
    X_test_tfidf  = tfidf.transform(X_test)         # transform test tweets using same vocabulary

    # Resampling to handle class imbalance
    # ros = RandomOverSampler(random_state=42)
    # X_train_tfidf, y_train = ros.fit_resample(X_train_tfidf, y_train)

    # param_grid = {
    #     'C': [0.01, 0.1, 1, 10],
    # }

    # grid = GridSearchCV(
    #     LogisticRegression(max_iter=1000, class_weight='balanced'),
    #     param_grid,
    #     cv=5,
    #     scoring='f1_weighted'
    # )

    # grid.fit(X_train_tfidf, y_train)
    # best_model = grid.best_estimator_

    # print("Best params:", grid.best_params_)

    # y_pred = best_model.predict(X_test_tfidf)
    # print(classification_report(y_test, y_pred, target_names=["negative", "neutral", "positive"]))

    # Models training
    models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Multinomial Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(class_weight='balanced')
    }

    # Run evaluation for all models
    for name, model in models.items():
        evaluate_model(name, model, X_train_tfidf, X_test_tfidf, y_train, y_test)

    # Save the best model
    best_model = models["Logistic Regression"]
    joblib.dump(best_model, "best_model.pkl")
    joblib.dump(tfidf, "tfidf_vectorizer.pkl")

    # Load the model and vectorizer
    best_model = joblib.load("best_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")

    # Predict sentiment of a new tweet
    tweet = "I didn't like the taxi driver that took me to the airport, and also I hate flying with this airline!"
    sentiment = predict_sentiment(tweet)
    print(f"Sentiment of the tweet: {sentiment}")

    tweet = "I didn't like the taxi driver that took me to the airport, but I love flying with this airline!"
    sentiment = predict_sentiment(tweet)
    print(f"Sentiment of the tweet: {sentiment}")

    tweet = "This airline, I saw you today."
    sentiment = predict_sentiment(tweet)
    print(f"Sentiment of the tweet: {sentiment}")