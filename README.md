# hate-speech-detection
Hate Speech and Offensive Language Detection using Machine Learning A Python-based NLP project that classifies tweets into three categories: Hate Speech, Offensive Language, or Neutral. Built using Scikit-Learn and NLTK with a Decision Tree Classifier.

This repository contains a Machine Learning pipeline designed to detect and classify toxic content in social media text. Using a dataset of approximately 24,783 tweets, the model categorizes text into three distinct labels: Hate Speech, Offensive Language, and No hate or offensive language.

🚀 Overview
The project implements a natural language processing (NLP) workflow that includes data cleaning, text preprocessing (stemming and stop-word removal), and a supervised learning model for multiclass classification.  

🛠️ Tech Stack
Language: Python  
Libraries:
pandas & numpy: Data manipulation and analysis.  
nltk: Natural Language Toolkit for text cleaning (Stopwords, Snowball/Porter Stemming).  
scikit-learn: Model building, training (DecisionTreeClassifier), and evaluation.  
seaborn & matplotlib: Data visualization and confusion matrix plotting.

📋 Features
Data Cleaning: Automatic removal of URLs, punctuation, numbers, and brackets.  
Text Transformation: Uses CountVectorizer to convert raw text into a sparse matrix for model training.  
Classification: Employs a Decision Tree algorithm to achieve high-accuracy labeling.  
Evaluation: Includes a confusion matrix and accuracy score metrics to validate model performance.

📊 Model Performance
The current model achieves an accuracy score of approximately 87.6%.

💻 How to Use
Requirements:
Bash
pip install pandas numpy scikit-learn nltk seaborn matplotlib
Run the Notebook:
Ensure twitter.csv is in the project directory.  
The model will preprocess the data, train on 67% of the dataset, and test on the remaining 33%.  
Predicting Custom Text:
Python
   # Example usage within the code:
   sample = "Your text here"
   cleaned_sample = clean_data(sample)
   vectorized_data = cv.transform([cleaned_sample]).toarray()
   prediction = dt.predict(vectorized_data)
   print(prediction)
📂 Dataset
The project utilizes a twitter.csv file containing the following columns:
tweet: The raw text content.  
class: Numerical labels (0: Hate Speech, 1: Offensive, 2: Neutral).
