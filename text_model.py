import numpy as np
import re
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class TextModel:
    def __init__(self, n=3, dimensions=1000):
        self.n = n
        self.dimensions = dimensions
        self.item_memory = None
        self.model = RidgeClassifier()

    def preprocess(self, text):
        text = re.sub(r'\d+\t|\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        return [text[i:i+self.n] for i in range(len(text)-self.n+1)]

    def initialize_item_memory(self, n_grams):
        self.item_memory = {n_gram: np.random.choice([-1, 1], self.dimensions) for n_gram in n_grams}

    def encode_text_to_hd_vector(self, text):
        ngrams = self.preprocess(text)
        hd_vector = np.zeros(self.dimensions)
        for ngram in ngrams:
            if ngram in self.item_memory:
                hd_vector += self.item_memory[ngram]
        return hd_vector

    def train_model(self, df):
        unique_n_grams = set()
        for sentence in df['Sentence']:
            unique_n_grams.update(self.preprocess(sentence))
        self.initialize_item_memory(unique_n_grams)
        X_hd = np.array([self.encode_text_to_hd_vector(sentence) for sentence in df['Sentence']])
        y = df['Language'].values
        X_train_hd, X_test_hd, y_train, y_test = train_test_split(X_hd, y, test_size=0.3, random_state=42)
        self.model.fit(X_train_hd, y_train)
        predictions = self.model.predict(X_test_hd)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Accuracy: {accuracy}")

    def predict_language(self, sentence):
        hd_vector = self.encode_text_to_hd_vector(sentence)
        hd_vector = hd_vector.reshape(1, -1)
        return self.model.predict(hd_vector)[0]