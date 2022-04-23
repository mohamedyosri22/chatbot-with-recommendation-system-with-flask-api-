import json
import numpy as np
from tensorflow import keras
from werkzeug.serving import WSGIRequestHandler
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from keras.preprocessing.text import Tokenizer
import random
import pickle
from flask import Flask, request, jsonify
import spacy

app = Flask(__name__)

stop_words = stopwords.words('english')

dataset = pd.read_csv('udemy_tech.csv')

model = keras.models.load_model('chat_model')

nlp = spacy.load('en_core_web_md')


with open('label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

dataset['Summary'] = dataset['Summary'].fillna('')
dataset['new'] = dataset['Title'] + ' ' + dataset['Summary']

import html


def remove_special_chars(text):
    re1 = re.compile(r'  +')
    x1 = text.lower().replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x1))


import unicodedata


def remove_non_ascii(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')


def to_lowercase(text):
    return text.lower()


import string


def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


def replace_numbers(text):
    return re.sub(r'\d+', '', text)


def remove_whitespaces(text):
    return text.strip()


def remove_stopwords(words, stop_words):
    return [word for word in words if word not in stop_words]


def stem_words(words):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in words]


from nltk.stem import WordNetLemmatizer


def lemmatize_words(words):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words]


def lemmatize_verbs(words):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word, pos='v') for word in words])


def text2words(text):
    return word_tokenize(text)


def normalize_text(text):
    text = remove_non_ascii(text)
    text = remove_punctuation(text)
    text = to_lowercase(text)
    text = replace_numbers(text)
    words = text2words(text)
    words = remove_stopwords(words, stop_words)
    words = lemmatize_words(words)
    words = lemmatize_verbs(words)

    return ''.join(words)


def normalize_corpus(corpus):
    return [normalize_text(t) for t in corpus]


def get_recommendations(title):
    title = [title]
    nor_new = normalize_corpus(dataset['new'])
    nor_input = normalize_corpus(title)
    tok = Tokenizer(num_words=10000, oov_token='UNK')
    tok.fit_on_texts(nor_new + nor_input)
    tfidf_ind = tok.texts_to_matrix(nor_new, mode='tfidf')
    tfidf_input = tok.texts_to_matrix(nor_input, mode='tfidf')

    cosine_sim = linear_kernel(tfidf_ind, tfidf_input)

    sim_scores = list(enumerate(cosine_sim))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[:3]

    courses_indices = [i[0] for i in sim_scores]

    return courses_indices


with open('intents.json') as file:
    data = json.load(file)

sims = []
sims_list = []

for intent in data['intents']:
    sims.append(intent['patterns'])

for sim in sims:
    for ans in sim:
        sims_list.append(ans.lower())


def reced(rec):
    re = ''
    rec = nlp(rec)
    rec2chat = ['recommend me a course', 'can you recommend me a course',
                'can you tell some course recommendations', "course recommendations", "course recommendations",
                "recommend me a course", "will you recommend me a course", "recommend course",
                "i need course recommendation","can you recommend courses","recommend courses","can you recommend courses","do you recommend courses"
                ,"courses recommendations"]


    sim_ratio = []
    for rec2 in rec2chat:
        sim_ratio.append(rec.similarity(nlp(rec2)))


    rec_ratio = np.max(sim_ratio)

    if rec_ratio > 0.93:
        course_field = request.json
        course_field = course_field["course_field"]
        reced_courses = get_recommendations(course_field)
        re = reced_courses

        return re
    else:
        return re


def ret_ans(userinp):

    userinp = remove_punctuation(userinp)

    max_len = 25

    reced_courses = ''
    output = []

    # recommendation function
    reced_courses = reced(userinp)
    if (len(reced_courses) != 0):
        for course in reced_courses:
            out = dataset.Title.iloc[course], "\nand the link : " + dataset.Link.iloc[course]
            output.append(out)
        return output

    else:

        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([userinp]),
                                                                          truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        for i in data['intents']:
            if i['tag'] == tag:
                out = np.random.choice(i['responses'])

        return out


@app.route('/', methods=['POST'])
def chat():
    userinp = request.json

    question = userinp['question']

    answer = ret_ans(question)

    return jsonify(answer)


@app.route('/')
def ind():
    return "<h1> chat bot api </h1>"


if __name__ == '__main__':
    WSGIRequestHandler.protocol_version = "HTTP/1.1"
    app.run(port=5050, debug=True)