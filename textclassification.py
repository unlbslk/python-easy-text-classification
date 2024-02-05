import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn import metrics
import joblib
import hashlib
import json
import csv

exampletext = "This is a good example!"

options =  """{
    "modelname": "textclassification",
    "hashname": "csvhashes",
    "checkCSVfiles": "true",
    "inputMode": "true",
    "enablePrints": "true",
    "stringJSONreply": "false"
}"""
parsedoptions = json.loads(options)


try: 
    model = joblib.load(f'{parsedoptions["modelname"]}.joblib')
    print("Checking model...") if parsedoptions["enablePrints"] == "true" else None
except FileNotFoundError:
    print("Training model...") if parsedoptions["enablePrints"] == "true" else None

if parsedoptions["checkCSVfiles"] == "true":
    with open("good_texts.csv", 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        row_count = sum(1 for row in csv_reader)
        if not row_count >= 2:
            raise ValueError("[ERROR] good_texts.csv must have 2 rows in it!")

    with open("bad_texts.csv", 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        row_count = sum(1 for row in csv_reader)
        if not row_count >= 2:
            raise ValueError("[ERROR] bad_texts.csv must have 2 rows in it!")

# Function to calculate the hash of the concatenated contents of CSV files
def calculate_data_hash():
    good_texts = pd.read_csv('good_texts.csv', names=['text'])
    bad_texts = pd.read_csv('bad_texts.csv', names=['text'])
    concatenated_data = pd.concat([good_texts, bad_texts])
    data_str = concatenated_data.to_string(index=False)
    data_hash = hashlib.md5(data_str.encode()).hexdigest()
    return data_hash

# Check if the data hash has changed
current_data_hash = calculate_data_hash()

try:
    stored_data_hash = joblib.load(f'{parsedoptions["hashname"]}.joblib')
except FileNotFoundError:
    stored_data_hash = None

if stored_data_hash != current_data_hash or not model:
    print("Retraining the model...") if parsedoptions["enablePrints"] == "true" else None
    good_texts = pd.read_csv('good_texts.csv', names=['text'])
    bad_texts = pd.read_csv('bad_texts.csv', names=['text'])
    good_texts['label'] = 'good'
    bad_texts['label'] = 'bad'
    data = pd.concat([good_texts, bad_texts])
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)
    joblib.dump(model, f'{parsedoptions["modelname"]}.joblib')
    joblib.dump(current_data_hash, f'{parsedoptions["hashname"]}.joblib')
    print("Training done!") if parsedoptions["enablePrints"] == "true" else None
else:
    print("Model loaded!") if parsedoptions["enablePrints"] == "true" else None

# Function to get user input
def get_user_input():
    new_text = str(input("> "))
    predicted_proba = model.predict_proba([new_text])
    predicted_label = model.predict([new_text])
    prob_good = predicted_proba[0][model.classes_ == 'good'][0]
    prob_bad = predicted_proba[0][model.classes_ == 'bad'][0]

    if parsedoptions["stringJSONreply"] == "false":
        print(f"Prediction: {predicted_label[0]}\nProbabilities: good: {prob_good*100:.2f}% | bad: {prob_bad*100:.2f}%")
    else:
        json_string = f'{{"prediction":"{predicted_label[0]}","good_prob":"{prob_good*100:.2f}%","bad_prob":"{prob_bad*100:.2f}%"}}'
        print(json_string)
    get_user_input()

def works_when_inputMode_is_not_true():
    predicted_proba = model.predict_proba([exampletext])
    predicted_label = model.predict([exampletext])
    prob_good = predicted_proba[0][model.classes_ == 'good'][0]
    prob_bad = predicted_proba[0][model.classes_ == 'bad'][0]
    if parsedoptions["stringJSONreply"] == "false":
        print(f"Prediction: {predicted_label[0]}\nProbabilities: good: {prob_good*100:.2f}% | bad: {prob_bad*100:.2f}%")
    else:
        json_string = f'{{"prediction":"{predicted_label[0]}","good_prob":"{prob_good*100:.2f}%","bad_prob":"{prob_bad*100:.2f}%"}}'
        print(json_string)

print("Ready!") if parsedoptions["enablePrints"] == "true" else None

# Checking user input mode
if parsedoptions["inputMode"] == "true": 
    get_user_input()
else:
    works_when_inputMode_is_not_true()
