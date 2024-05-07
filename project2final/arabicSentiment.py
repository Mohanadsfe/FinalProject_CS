import re
import nltk
import requests
import pandas as pd
import zipfile
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

nltk.download('stopwords')
nltk.download('punkt')

# URLs of the zip files
urls = ['http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/EI-reg/Arabic/2018-EI-reg-Ar-train.zip',
        'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/EI-reg/Arabic/2018-EI-reg-Ar-train.zip',
         'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/EI-reg/Arabic/2018-EI-reg-Ar-dev.zip']

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

import langdetect
from googletrans import Translator

# Detect and translate non-English tweets to English
def detect_and_translate_to_english(tweet):
    try:
        lang = langdetect.detect(tweet)
        if lang != 'en':
            translator = Translator()
            translation = translator.translate(tweet, src=lang, dest='en')
            tweet = translation.text
        return tweet
    except:
        return tweet
        
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

            combined_data = combined_data._append(data, ignore_index=True)


print("Combined Data:")
print(combined_data.head())  # Print the first few rows of the combined data
combined_data.to_csv("modified_dataset_arabic.csv", index=False)


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


# Clean the tweet text
import string
# Define Arabic stopwords as a list of strings
arabic_stopwords = stopwords.words('arabic')

# Custom function for text preprocessing
def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # Convert text to lowercase
    text = text.lower()
    # Tokenize text and remove stopwords
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in arabic_stopwords]
    # Join tokens back into text
    text = ' '.join(tokens)
    return text



# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(combined_data['Affect Dimension'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(combined_data['Tweet'], y_encoded, test_size=0.2, random_state=42)


# Create a pipeline for each classifier
def create_pipeline(clf):
    return Pipeline([
        ('tfidf', TfidfVectorizer(sublinear_tf=True, norm='l2', encoding='latin-1', ngram_range=(1, 2), preprocessor=preprocess_text, max_features=5000)),
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
import joblib
# Function to perform grid search for a classifier
def grid_search_classifier(classifier_name, classifier_obj, parameters):
    pipeline = create_pipeline(classifier_obj)
    grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    joblib.dump(grid_search.best_estimator_, f'{classifier_name}_best_model.pkl')
    return classifier_name, grid_search.best_params_, grid_search.best_score_

import concurrent
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


for classifier_name in classifiers:
    model = joblib.load(f'{classifier_name}_best_model.pkl')
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    # Visualize confusion matrix using a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f"Confusion Matrix for {classifier_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

# Load the best model (you can choose the model you want to use)
selected_model = 'SVC'  # Change this to the model you want to use
best_model = joblib.load(f'{selected_model}_best_model.pkl')

prdct_list = ["sadnees","joy","fear","anger"]

# Function to predict sentiment for user input
def predict_sentiment_for_user_input(model, label_encoder):
    user_input = input("Enter your tweet: ")
    print(user_input)
    user_input = detect_and_translate_to_english(user_input)
    print(user_input)

    # Clean user input
    # Make predictions
    prediction = model.predict([user_input])
    # Decode the predicted label
    predicted_sentiment = label_encoder.inverse_transform(prediction)[0]
    print("predicted_sentiment" + str(predicted_sentiment))
    print(f"Predicted sentiment:{prdct_list[predicted_sentiment]}")

x = 1
while x:
    predict_sentiment_for_user_input(best_model, label_encoder)
    print("\n")
