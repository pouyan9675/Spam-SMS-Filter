#!/usr/bin/python3
# -*- coding: utf-8 -*-
from pathlib import Path
from hazm import *
import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import svm
from sklearn import metrics

email_regex = "^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$"
phone_regex = "((\+98|0)?9\d{9})|(0\d{2}\d{8}|\d{8})"
number_regex = "^[1-9]\d*$"
url_regex = "(@^(https?|ftp)://[^\s/$.?#].[^\s]*$@iS)|(t.me/[a-z|0-9]{4,})"\
    "|((https?://)?(w{3}.)?[a-zA-Z0-9]+.[a-zA-Z]{2,}(/[a-zA-Z0-9]*)*)"


def numbers_to_english(text):
    text = text.replace('۰', '0')
    text = text.replace('۱', '1')
    text = text.replace('۲', '2')
    text = text.replace('۳', '3')
    text = text.replace('۴', '4')
    text = text.replace('۵', '5')
    text = text.replace('۶', '6')
    text = text.replace('۷', '7')
    text = text.replace('۸', '8')
    text = text.replace('۹', '9')
    return text


def preprocessing_text(text):
    text = numbers_to_english(text)                 # to work with regex
    text = re.sub(email_regex, 'آدرس_ایمیل', text)  # Replace email addresses
    text = re.sub(phone_regex, 'شماره_تلفن', text)  # Replace phone numbers
    text = re.sub(url_regex, 'آدرس_لینک', text)     # Replace urls
    text = re.sub(number_regex, 'عدد_رقم', text)     # Replace numbers
    normalizer = Normalizer()
    text = normalizer.normalize(text)
    tokens = word_tokenize(text)         # Tokenization
    stemmer = Stemmer()
    for index, term in enumerate(tokens):
        tokens[index] = stemmer.stem(term)  # Stemming
    text = ' '.join(term for term in tokens if term not in stop_words.values)     # Removing stop words
    return text


def detect_spam():
    test_file = Path('manual_test.txt')
    if test_file.is_file():
        with open('manual_test.txt', 'r', encoding='utf-8', errors='ignore') as message:
            sms = message.read()
            features = vectorizer.transform([preprocessing_text(sms)])
            result = classifier.predict(features)
            print(sms + '\n')
            if result:
                print('Detect As : Spam')
            else:
                print('Detect As : Not Spam')
    else:
        print('No test file!')


if __name__ == '__main__':
    documents = []

    df = pd.read_csv('Dataset.csv', encoding='utf-8')
    raw_messages = df['Message']

    le = LabelEncoder()
    labels = le.fit_transform(df['Tag'])

    stop_words = pd.read_csv('stop-words.txt', encoding='utf-8', delimiter='\n', header=None)

    for content in raw_messages:
        content = preprocessing_text(content)
        documents.append(content)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))    # Building unigram and bigrams
    n_grams = vectorizer.fit_transform(documents)       # Counting unigram , bigrams and then calculate tf-idf

    
    # X_train, X_test, y_train, y_test = train_test_split(
    #     n_grams,
    #     labels,
    #     test_size=0.3,      # test set 20% of whole data
    #     random_state=42,
    #     shuffle=True,
    #     stratify=labels
    # )

    classifier = svm.LinearSVC(loss='hinge', C=1)
    scores = cross_val_score(classifier,
        n_grams,
        labels,
        cv=StratifiedShuffleSplit(n_splits=10, test_size=0.2),
        scoring='f1')

    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # classifier.fit(X_train, y_train)                                # Testing with hold-out method
    # y_pred = classifier.predict(X_test)
    # f_measure = metrics.f1_score(y_test, y_pred)                    # Calculating F1 Measure
    #
    # confusion_matrix = pd.DataFrame(
    #     metrics.confusion_matrix(y_test, y_pred),
    #     index=[['actual', 'actual'], ['spam', 'ham']],
    #     columns=[['predicted', 'predicted'], ['spam', 'ham']]
    # )
    #
    # print('F1 Score : ' + str(f_measure))
    # print(confusion_matrix)

    print('\n============= Common Spam words =============')
    classifier.fit(n_grams, labels)
    common_spams = pd.Series(
        classifier.coef_.T.ravel(),                 # classifier.features_weight.shape_log_probabilities(n_samples,n_classes).flatted_array
        index=vectorizer.get_feature_names()
    ).sort_values(ascending=False)[:20]
    print(common_spams)

    print('\n============= Manual Test Set =============')
    classifier.fit(n_grams, labels)
    detect_spam()
