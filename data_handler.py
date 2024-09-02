import os
import pandas as pd

# Constants
language_labels = {
    'Romanian_2015_10K': 'Romanian',
    'Swedish_2007_10K': 'Swedish',
    'Lithuanian_2015_10K': 'Lithuanian',
    'Slovak_2020_10K': 'Slovak',
    'Slovene_2019_10K': 'Slovene',
    'Polish_2007_10K': 'Polish',
    'Italian_2005-2009_10K': 'Italian',
    'Portuguese_2019_10K': 'Portuguese',
    'Latvian_2019_10K': 'Latvian',
    'Spanish_2006_10K': 'Spanish',
    'French_2005-2008_10K': 'French',
    'Estonia_2014_10K': 'Estonian',
    'German_2007_10K': 'German',
    'Bulgarian_2007_10K': 'Bulgarian',
    'Danish_2007_10K': 'Danish',
    'English_2007_10K': 'English',
    'Dutch_2016_10K': 'Dutch',
    'Hungarian_2019_10K': 'Hungarian',
    'Finnish_2005-2007_10K': 'Finnish',
    'Czech_2005-2007_10K': 'Czech'
}

class DataHandler:
    def __init__(self, base_path):
        self.base_path = base_path
        self.language_texts = {}

    def find_sentence_files(self, directory):
        sentence_files = []
        for item in os.listdir(directory):
            path = os.path.join(directory, item)
            if os.path.isdir(path):
                sentence_files.extend(self.find_sentence_files(path))
            elif '-sentences.txt' in item:
                sentence_files.append(path)
        return sentence_files

    def load_sentences(self, files):
        all_texts = []
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    all_texts.extend(file.read().strip().split('\n'))
            except UnicodeDecodeError:
                print(f"Encoding error in file {file_path}")
            except IOError:
                print(f"Could not read file {file_path}")
        return all_texts

    def load_data(self):
        if not os.path.exists(self.base_path):
            print(f"Base path {self.base_path} does not exist.")
            return

        for language_dir in os.listdir(self.base_path):
            dir_path = os.path.join(self.base_path, language_dir)
            if os.path.isdir(dir_path):
                sentence_files = self.find_sentence_files(dir_path)
                self.language_texts[language_dir] = self.load_sentences(sentence_files)

        # Update keys to use simple language names
        self.language_texts = {language_labels[key]: value for key, value in self.language_texts.items() if key in language_labels}

    def create_dataframe(self):
        data = []
        for language, sentences in self.language_texts.items():
            for sentence in sentences:
                data.append({'Language': language, 'Sentence': sentence})
        return pd.DataFrame(data)