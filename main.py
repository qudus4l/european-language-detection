from data_handler import DataHandler
from text_model import TextModel

BASE_PATH = '/Users/Q/Downloads/european-language-detection/Ridge'

if __name__ == '__main__':
    handler = DataHandler(BASE_PATH)
    handler.load_data()
    df = handler.create_dataframe()

    text_model = TextModel()
    text_model.train_model(df)

    while True:
        sentence = input("Enter a sentence to predict its language or type 'exit' to quit: ")
        if sentence.lower() == 'exit':
            break
        language = text_model.predict_language(sentence)
        print(f"The predicted language is: {language}")