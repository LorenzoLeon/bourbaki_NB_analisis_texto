# Regex and standarization
import re
import unidecode
# extra Utils
import random
# Dataframes
import pickle
import pandas as pd
# Visualización
import matplotlib.pyplot as plt
import matplotlib
from wordcloud import WordCloud
# Natural Language tk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk import FreqDist, NaiveBayesClassifier, classify

stop_words_sp = set(stopwords.words('spanish'))
stop_words_sp = stop_words_sp.union(
    [".", ",", ")", "(", "'", ":", "?", "¿", "!", "¡", " ", "@", "si", "#", "...", "https", "http", "tco", "  "])


def deEmojify(text):
    regrex_pattern = re.compile(pattern="["
                                        u"\U0001F600-\U0001F64F"  # emoticons
                                        u"\U0001F300-\U0001F5FF"  # símbolos & pictogramas
                                        u"\U0001F680-\U0001F6FF"  # transporte & símbolos mapas
                                        u"\U0001F1E0-\U0001F1FF"  # banderas0 (iOS)
                                        "]+", flags=re.UNICODE)
    return regrex_pattern.sub(r'emoji', text)


def create_word_cloud(base, idgraf):
    # hacer el análisis de WordCloud
    wordcloud = WordCloud(background_color="white", stopwords=stop_words_sp, random_state=2016).generate(
        " ".join([i for i in base['text']]))
    # Preparar la figura
    plt.figure(num=idgraf, figsize=(5, 2), facecolor='k', dpi=5000)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Good Morning Datascience+")


def open_pickle(name="tweets_covid.pickle"):
    # Open Classifier
    file_to_open = open(name, "rb")
    object_opened = pickle.load(file_to_open)
    file_to_open.close()
    return object_opened


def create_corpus():
    tweets_covid = pd.read_csv("tweets_covid.csv")
    create_word_cloud(tweets_covid[tweets_covid['type'] == "positive"], "positivo")
    create_word_cloud(tweets_covid[tweets_covid['type'] == "negative"], "negativo")
    plt.show()

    tweets_covid = pd.read_csv("tweets_covid.csv")

    tweets_clean = tweets_covid.groupby("type").sample(n=110000, random_state=1)
    tweets_covid.to_pickle("tweets_covid.pickle", protocol=4)
    tweets_clean.to_pickle("tweets_covid_balanced.pickle", protocol=4)
    (tweets_covid.groupby("type").sample(n=11000, random_state=1)).to_pickle("tweets_small.pickle", protocol=3)


if __name__ == '__main__':
    # Leer los archivos de texto como un corpus, ya fueron guardados en formato pickle.
    # open_pickle()
    tweets_covid = open_pickle("tweets_clean.pickle")

    print(tweets_covid["text"][62527])
    word_tokens = word_tokenize(tweets_covid["text"][678023], language="spanish")
    print(word_tokens)
    filtered_sentence = [w for w in word_tokens if not w in stop_words_sp]
    print(filtered_sentence)

    tweets_covid['text'] = tweets_covid['text'].apply(lambda x: re.sub('[¡!@#$:).;,¿`?&]', '', x.lower()))
    tweets_covid['text'] = tweets_covid['text'].apply(lambda x: re.sub('  ', ' ', x))

    create_word_cloud(tweets_covid[tweets_covid['type'] == "positive"], "positivo")
    plt.show()

    create_word_cloud(tweets_covid[tweets_covid['type'] == "negative"], "negativo")
    plt.show()

    # Normal Classifier
    # CREATE CLASSIFFIER NORMAL
    # CREATE CLASSIFFIER NORMAL
    all_words = []
    for row in tweets_covid["content"]:
        # for w in word_tokenize(unidecode.unidecode(row), language="spanish"): # quitando acentos
        for w in word_tokenize(row, language="spanish"):  # sin quitar acentos
            if w not in stop_words_sp:
                all_words.append(w.lower())

    all_words = FreqDist(all_words)
    print(all_words.most_common(15))
    print(all_words["película"])

    word_features = list(all_words.keys())[:3000]

    documents = [(list(word_tokenize(text, language="spanish")), sentiment)
                 for sentiment, text in zip(tweets_covid["type"], tweets_covid["text"])]

    random.shuffle(documents)


    def find_features(document):
        words = set(document)
        features = {}
        for w in word_features:
            features[w] = (w in words)
        return features


    print((find_features(my_corpus.words('positivo_991.txt'))))
    featuresets = [(find_features(rev), category) for (rev, category) in documents]
    # set that we'll train our classifier with
    training_set = featuresets[:3100]
    # set that we'll test against.
    testing_set = featuresets[3100:]
    classifier = NaiveBayesClassifier.train(training_set)
    print("Classifier accuracy percent:", (classify.accuracy(classifier, testing_set)) * 100)
    classifier.show_most_informative_features(15)

    # Stemming Classifier
    # FOR STEMMING
    stemmer = SnowballStemmer('spanish')
    # example
    # for w in word_tokens:
    #    print(stemmer.stem(w))
    all_words_stem = []
    for www in my_corpus.words():
        stemmed_word = stemmer.stem(www)
        if stemmed_word not in stop_words_sp:
            all_words_stem.append(stemmed_word)
    all_words_stem = FreqDist(all_words_stem)

    print(all_words_stem.most_common(15))
    # print(all_words_stem["pelicula"])
    word_features_stem = list(all_words_stem.keys())[:3000]


    def find_features_stem(document):
        words = set(document)
        features = {}
        for w in word_features_stem:
            features[w] = (w in words)
        return features


    # print((find_features_stem(corpus.words('positivo_991.txt'))))
    feature_sets_stem = [(find_features_stem(rev), category) for (rev, category) in documents]
    training_set_stem = feature_sets_stem[:3100]
    # set that we'll test against.
    testing_set_stem = feature_sets_stem[3100:]
    # classifier.classify(find_features_stem(my_corpus.words('positivo_991.txt')))
    # classifier.classify(find_features_stem(my_corpus.words('negativo_1001.txt')))
    classifier = NaiveBayesClassifier.train(training_set_stem)
    print("Classifier accuracy percent:", (classify.accuracy(classifier, testing_set_stem)) * 100)
    classifier.show_most_informative_features(15)
    print("Classifier accuracy percent:", (classify.accuracy(classifier, testing_set_stem)) * 100)
    classifier.show_most_informative_features(15)

    # No accents classifier
    # No accents:
    all_words = []
    for www in my_corpus.words():
        sin_acc_w = unidecode.unidecode(www)
        if sin_acc_w not in stop_words_sp:
            all_words.append(sin_acc_w.lower())
    all_words = FreqDist(all_words)
    # print(all_words.most_common(15))
    # print(all_words["pelicula"])
    word_features = list(all_words.keys())[:3000]


    def find_features(document):
        words = set(document)
        features = {}
        for w in word_features:
            features[w] = (w in words)
        return features


    # print((find_features(corpus.words('positivo_991.txt'))))
    featuresets = [(find_features(rev), category) for (rev, category) in documents]
    # set that we'll train our classifier with
    training_set = featuresets[:3100]
    # set that we'll test against.
    testing_set = featuresets[3100:]
    classifier = NaiveBayesClassifier.train(training_set)
    print("Classifier accuracy percent:", (classify.accuracy(classifier, testing_set)) * 100)
    classifier.show_most_informative_features(15)
