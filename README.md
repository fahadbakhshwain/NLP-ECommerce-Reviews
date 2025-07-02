# E-commerce Customer Review Analysis using NLP

## 1. Project Overview
This project leverages Natural Language Processing (NLP) to perform sentiment analysis and topic modeling on a large dataset of e-commerce customer reviews. The goal is to automate the process of understanding customer feedback, identifying key issues, and classifying opinions to drive business improvements.

## 2. Business Problem
In a competitive e-commerce landscape, understanding customer feedback at scale is crucial. Manually reading thousands of reviews is inefficient and prone to bias. This project addresses the need for an automated system that can accurately classify review sentiment (positive/negative) and extract the core topics customers are discussing, allowing the business to quickly identify product strengths, weaknesses, and areas for operational improvement.

## 3. Data Source
The data was sourced from a public dataset on **[اذكر هنا المصدر، مثلاً: Kaggle]**. It includes the raw text of customer reviews and their corresponding star ratings, which were used to label the sentiment.

## 4. Methodology

1.  **Data Cleaning & Preprocessing:**
    * Standardized text to lowercase.
    * Removed punctuation, URLs, and special characters.
    * Performed tokenization to split reviews into individual words.
    * Removed common English stopwords.
    * Applied lemmatization using NLTK to reduce words to their base form (e.g., "running" -> "run").

2.  **Exploratory Data Analysis (EDA):**
    * Generated word clouds to visualize the most frequent terms in positive vs. negative reviews.
    * Analyzed the distribution of review lengths to identify patterns.

3.  **Feature Engineering:**
    * Transformed the cleaned text into numerical features using the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer, which gives more weight to words that are important to a specific review.

4.  **Modeling & Evaluation:**
    * **Sentiment Analysis:** Trained and evaluated three different machine learning models (Logistic Regression, Naive Bayes, SVM) to classify reviews. The dataset was split into training (80%) and testing (20%) sets.
    * **Model Evaluation:** Performance was measured using Accuracy, Precision, Recall, and F1-Score. The confusion matrix was analyzed to understand classification errors.

## 5. Key Findings & Results
* The **Logistic Regression model** provided the best balance of performance, achieving an **F1-Score of [0.91]** on the test set.
* Topic modeling using LDA identified **[3]** primary themes in negative reviews: **['Shipping & Delivery Issues', 'Product Quality & Defects', and 'Incorrect Sizing']**.
* The analysis confirmed that reviews mentioning words like "late," "broken," or "wrong size" were highly correlated with 1- and 2-star ratings.

## 6. Technologies Used
- **Language:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, NLTK, Scikit-learn
- **Environment:** Jupyter Notebook

## 7. How to Run
1.  Clone the repository: `git clone [ضع رابط المستودع الخاص بك هنا]`
2.  Install the necessary libraries: `pip install -r requirements.txt` (إذا كان لديك ملف requirements.txt)
3.  Open and run the `[اكتب اسم ملف النوت بوك هنا].ipynb` notebook in a Jupyter environment.
