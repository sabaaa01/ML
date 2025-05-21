# 📊 Sentiment Analysis using Random Forest and Naive Bayes

This project performs sentiment analysis on product reviews using machine learning algorithms — **Naive Bayes** and **Random Forest**. It classifies textual reviews into **positive** or **negative** sentiments based on natural language processing (NLP) and supervised learning.

---

## 🧠 Algorithms Used

- **Naive Bayes Classifier:** A probabilistic model based on Bayes’ Theorem, efficient for text classification.
- **Random Forest Classifier:** An ensemble model that builds multiple decision trees and combines their output for better accuracy and robustness.

---

## 🛠️ Features

- Preprocessing of textual data (cleaning, tokenization, stopword removal, lemmatization)
- Feature extraction using **TF-IDF**
- Model training and evaluation
- Comparison of Naive Bayes and Random Forest performance
- Performance metrics: Accuracy, Precision, Recall, F1-score

---

## 📁 Dataset

- **Source:** Kaggle
- **Attributes:** `Review Text`, `Sentiment Label (Positive/Negative)`

---

## 🔧 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- NLTK
- Matplotlib / Seaborn (optional for visualization)

---

## 🧹 Text Preprocessing Steps

1. Convert text to lowercase
2. Remove punctuation and special characters
3. Tokenize text into words
4. Remove stopwords
5. Lemmatize words
6. Apply **TF-IDF** vectorization

---

## 🚀 How to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis-ml.git
   cd sentiment-analysis-ml
