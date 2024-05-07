import requests
import pandas as pd
import zipfile
import os
import joblib
import concurrent.futures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from pyarabic.araby import strip_tashkeel, strip_tatweel
from pyarabic.araby import tokenize as arabic_tokenize

# Function to strip diacritic signs from Arabic words
def remove_diacritics(word):
    return strip_tashkeel(word)

# Function to convert Arabic words to their base forms
def convert_to_base_form(text):
    tokens = arabic_tokenize(text)
    base_forms = []
    for token in tokens:
        base_form = strip_tashkeel(strip_tatweel(token))
        base_forms.append(base_form)
    return ' '.join(base_forms)

# Load the synonym dataset
synonym_df = pd.read_csv('sameMeaningWords.csv')

# Function to classify words based on dialectical context
def classify_dialectical_sentiment(word, dialect):
    # Your implementation to classify sentiment based on dialect
    pass

# Function to replace words in tweets with their synonyms from the CSV file
def replace_synonyms(tweet, synonym_df, city):
    words = tweet.split()
    replaced_tweet = []
    for word in words:
        synonyms = synonym_df.loc[synonym_df['Word'] == word, 'Synonym1':'Synonym5'].values.flatten()
        if len(synonyms) > 0:
            # Get the sentiment for each synonym based on the dialectical context (city)
            synonym_sentiments = [classify_dialectical_sentiment(synonym, city) for synonym in synonyms]
            # Choose the synonym with the most frequent sentiment
            chosen_synonym = max(set(synonym_sentiments), key=synonym_sentiments.count)
            replaced_tweet.append(chosen_synonym)  # Replace with the chosen synonym
        else:
            replaced_tweet.append(word)
    return ' '.join(replaced_tweet)

# URLs of the zip files
urls = ['http://www.saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/EI-reg/English/EI-reg-En-train.zip']

# Folder for downloaded zip files and extracted data
download_folder = 'downloaded_zips'
extract_folder = 'extracted_data'

# Create folders if they don't exist
os.makedirs(download_folder, exist_ok=True)
os.makedirs(extract_folder, exist_ok=True)

def is_zipfile(filename):
    try:
        _ = zipfile.ZipFile(filename)
        return True
    except zipfile.BadZipFile:
        return False

# Headers to mimic a browser request
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

# Download and extract each zip file
for url in urls:
    try:
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()  # Check if the download was successful

        zip_file_name = os.path.join(download_folder, url.split('/')[-1])

        with open(zip_file_name, 'wb') as file:
            file.write(response.content)

        if is_zipfile(zip_file_name):
            with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
                zip_ref.extractall(extract_folder)
        else:
            print(f"Downloaded file is not a zip file: {zip_file_name}")

    except requests.HTTPError as e:
        print(f"HTTP Error occurred while downloading {url}: {e}")
    except Exception as e:
        print(f"An error occurred while downloading {url}: {e}")

# Combine all datasets, only keeping 'Tweet Affect' and 'Dimension' columns
combined_data = pd.DataFrame()
for root, dirs, files in os.walk(extract_folder):
    for file in files:
        if file.endswith('.txt'):
            file_path = os.path.join(root, file)
            # Adjust the delimiter and column names/indices as per your file format
            data = pd.read_csv(file_path, delimiter='\t', usecols=['Tweet', 'Affect Dimension'])

            print(f"Data from file {file_path}:")
            print(data.head())  # Print the first few rows of data

            combined_data = pd.concat([combined_data, data], ignore_index=True)

print("Combined Data:")
print(combined_data.head())  # Print the first few rows of the combined data

# Apply preprocessing: Convert Arabic words to their base forms
combined_data['Tweet'] = combined_data['Tweet'].apply(convert_to_base_form)

# Apply diacritic manipulation to Arabic tweets
combined_data['Tweet'] = combined_data['Tweet'].apply(remove_diacritics)

# Replace words with synonyms and update sentiment class
combined_data['Tweet'] = combined_data.apply(lambda row: replace_synonyms(row['Tweet'], synonym_df, city='Haifa'), axis=1)

# Function to check synonyms and update sentiment class
def check_synonyms(tweet, synonym_df, current_sentiment):
    words = tweet.split()
    for word in words:
        synonyms = list(synonym_df[synonym_df['Word'] == word].iloc[:, 1:].values.flatten())
        for synonym in synonyms:
            if synonym in synonym_df['Word'].values:
                if synonym_df[synonym_df['Word'] == synonym]['Sentiment'].values[0] != current_sentiment:
                    return False
    return True

# Apply double testing for neutral words and update sentiment class
combined_data['Affect Dimension'] = combined_data.apply(lambda row: row['Affect Dimension'] if check_synonyms(row['Tweet'], synonym_df, row['Affect Dimension']) else 'positive', axis=1)

# Count the number of tweets for each sentiment class
sentiment_counts = combined_data['Affect Dimension'].value_counts()

# Print the counts for each sentiment class
for sentiment, count in sentiment_counts.items():
    print(f"Number of tweets with sentiment '{sentiment}': {count}")

def classifiers_func(affect):
    if affect == 'sadness':
        return 0
    elif affect == 'joy':
        return 1
    elif affect == 'fear':
        return 2
    else:
        return 3

combined_data['Affect Dimension'] = combined_data['Affect Dimension'].apply(classifiers_func)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(combined_data['Affect Dimension'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(combined_data['Tweet'], y_encoded, test_size=0.2, random_state=42)

# Create a pipeline for each classifier
def create_pipeline(clf):
    return Pipeline([
        ('tfidf', TfidfVectorizer(sublinear_tf=True, norm='l2', encoding='latin-1', stop_words='english', max_features=5000)),
        ('classifier', clf),
    ])

# Define classifiers
classifiers = {
    'LogisticRegression': LogisticRegression(random_state=42),
    'MultinomialNB': MultinomialNB(),
    'RandomForestClassifier': RandomForestClassifier(random_state=42),
    'SVC': SVC(random_state=42),
    'KNeighborsClassifier': KNeighborsClassifier(),
}

# Define hyperparameters for each classifier
hyperparameters = {
    'LogisticRegression': {'classifier__C': [0.001, 0.01, 0.1, 1, 10], 'classifier__solver': ['lbfgs', 'saga']},
    'MultinomialNB': {'classifier__alpha': [0.1, 0.5, 1, 1.5, 2]},
    'RandomForestClassifier': {'classifier__n_estimators': [10, 50, 100, 200], 'classifier__max_depth': [10, 50, 100, None]},
    'SVC': {'classifier__C': [0.1, 1, 10], 'classifier__kernel': ['linear', 'rbf']},
    'KNeighborsClassifier': {'classifier__n_neighbors': [3, 5, 7, 9], 'classifier__weights': ['uniform', 'distance']},
}

# Function to perform grid search for a classifier
def grid_search_classifier(classifier_name, classifier_obj, parameters):
    pipeline = create_pipeline(classifier_obj)
    grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    joblib.dump(grid_search.best_estimator_, f'{classifier_name}_best_model.pkl')
    return classifier_name, grid_search.best_params_, grid_search.best_score_

# Parallelize grid search using ThreadPoolExecutor
best_parameters = {}
with concurrent.futures.ThreadPoolExecutor(max_workers=len(classifiers)) as executor:
    future_to_classifier = {executor.submit(grid_search_classifier, name, clf, hyperparameters.get(name, {})): name for name, clf in classifiers.items()}
    for future in concurrent.futures.as_completed(future_to_classifier):
        classifier_name = future_to_classifier[future]
        classifier_name, params, score = future.result()
        best_parameters[classifier_name] = {'parameters': params, 'score': score}

# Print best parameters and scores for each classifier
for classifier_name, info in best_parameters.items():
    print(f"Best parameters for {classifier_name}: {info['parameters']}")
    print(f"Score: {info['score']}")

# Evaluate and compare models
metrics = {}
for classifier_name in classifiers:
    model = joblib.load(f'{classifier_name}_best_model.pkl')
    y_pred = model.predict(X_test)
    metrics[classifier_name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1 Score': f1_score(y_test, y_pred, average='weighted')
    }

# Convert the metrics dictionary to a DataFrame
metrics_df = pd.DataFrame(metrics).transpose()

# Plot charts comparing all metrics of the best models
metrics_df.plot(kind='bar', figsize=(12, 8))
plt.title("Comparison of Metrics for Best Models")
plt.xlabel("Classifiers")
plt.ylabel("Score")
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Evaluate and compare models with different techniques
metrics_techniques = {}
for classifier_name, clf in classifiers.items():
    for technique in ['TF-IDF', 'n-grams']:
        pipeline = create_pipeline(clf)
        if technique == 'TF-IDF':
            pipeline.named_steps['tfidf'].set_params(ngram_range=(1, 1))  # For unigrams
        elif technique == 'n-grams':
            pipeline.named_steps['tfidf'].set_params(ngram_range=(1, 2))  # For unigrams and bigrams
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        metrics_techniques.setdefault((classifier_name, technique), {}).update({
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        })

# Convert the metrics dictionary to a DataFrame
metrics_techniques_df = pd.DataFrame(metrics_techniques).transpose()

# Plot charts comparing all metrics of the best models with different techniques
metrics_techniques_df.plot(kind='bar', figsize=(15, 10))
plt.title("Comparison of Metrics for Best Models with Different Techniques")
plt.xlabel("Classifiers and Techniques")
plt.ylabel("Score")
plt.legend(loc='best')
plt.tight_layout()
plt.show()
