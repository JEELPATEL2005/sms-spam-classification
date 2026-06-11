# sms-spam-classification


# SMS Spam Classification

## Project Overview
This project builds a machine learning model to classify SMS messages as either **spam** or **ham** (legitimate messages). Using natural language processing (NLP) techniques and various Naive Bayes classifiers, the model achieves high accuracy in identifying unwanted spam messages.


## Project Structure

### 1. Data Loading & Exploration
- Load SMS data using pandas
- Label encoding: Convert "ham" (0) and "spam" (1) to numeric values
- Check for missing values and duplicates
- Exploratory Data Analysis (EDA) with visualizations

### 2. Data Preprocessing
- **Lowercasing:** Convert all text to lowercase
- **Tokenization:** Split text into individual words using NLTK
- **Alphanumeric Filtering:** Remove special characters
- **Stopword Removal:** Eliminate common English stopwords
- **Stemming:** Apply Porter Stemmer to reduce words to root forms

### 3. Feature Engineering
- **TF-IDF Vectorization:** Convert text to numerical features using `TfidfVectorizer` with max 3,000 features
- **Character Count:** Calculate message length
- **Word Count:** Count tokens per message

### 4. Model Building & Evaluation
Three Naive Bayes classifiers were tested:
- **Gaussian Naive Bayes (GNB)**
- **Multinomial Naive Bayes (MNB)** ⭐ *Selected Model*
- **Bernoulli Naive Bayes (BNB)**

### 5. Model Persistence
- Vectorizer saved as `vectorizer.pkl`
- Trained model saved as `model.pkl`

## Performance Metrics
- **Accuracy:** Measures overall correctness
- **Precision:** Measures accuracy of spam predictions
- **Recall:** Measures ability to identify all spam messages
- **Confusion Matrix:** Shows true positives, false positives, true negatives, false negatives

## Dependencies
```
pandas
numpy
scikit-learn
nltk
matplotlib
seaborn
```

## Installation
```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn
```

## Usage

### Training the Model
Run the Jupyter Notebook cells in sequence to:
1. Load and explore the data
2. Preprocess text data
3. Train and evaluate models
4. Save the vectorizer and model

## Files
- `sms_spam_classification.ipynb` - Main notebook with full pipeline
- `spam.csv` - Dataset
- `vectorizer.pkl` - Saved TF-IDF vectorizer
- `model.pkl` - Saved trained model

## Future Improvements
- Experiment with advanced models (SVM, Random Forest, Deep Learning)
- Implement cross-validation for better generalization
- Add more sophisticated NLP techniques (word embeddings, BERT)
- Create a web interface for real-time predictions
- Handle class imbalance with SMOTE or class weights

