
# European Language Detection

This repository contains the code and resources for a machine learning project aimed at recognizing the language of input texts from 21 European languages. The project uses distributed high-dimensional representations and Ridge regression to classify texts based on their language.

## Languages Covered
The model recognizes the following languages:
- Bulgarian, Czech, Danish, German, Greek, English, Estonian, Finnish, French, Hungarian, Italian, Latvian, Lithuanian, Dutch, Polish, Portuguese, Romanian, Slovak, Slovene, Spanish, Swedish.

## Data Source
The training and testing data are sourced from the Wortschatz Corpora provided by the University of Leipzig. You can download the datasets [here](https://wortschatz.uni-leipzig.de/en/download).

## Getting Started

### Prerequisites
- Python 3.6+
- Libraries: numpy, pandas, scikit-learn
  ```bash
  pip install numpy pandas scikit-learn
  ```

### Installation
Clone this repository to your local machine:
```bash
git clone https://github.com/qudus4l/european-language-detection.git
cd european-language-detection
```

### Usage
Run the main script to start the language detection:
```bash
python main.py
```
Follow the on-screen prompts to input sentences and get language predictions. It might take a while to come up as model is training

## Methodology

### Data Preprocessing
Texts are preprocessed to remove digits, punctuation, and other non-textual elements, followed by conversion into lowercase for uniformity.

### Feature Encoding
- **N-grams**: Uses tri-grams (n=3) representation.
- **Encoding**: Each n-gram is encoded into d-dimensional {+1, -1} vectors. Vector lengths tested include 100, 1000, and 10000 dimensions.

### Model Training
A Ridge regression model from scikit-learn is trained on 70% of the data, with the remaining 30% used for testing.

### Testing and Evaluation
The test phase involves encoding unknown text samples and predicting their languages using the trained model. Performance is evaluated based on accuracy, F1-score, and a confusion matrix.

## Results
Include any significant results, performance metrics, or graphs here.

## References
- [Wortschatz Corpora Data](https://wortschatz.uni-leipzig.de/en/download)
- [Research on n-gram Statistics and High-dimensional Computing](https://www.researchgate.net/publication/337293395_Distributed_Representation_of_n-gram_Statistics_for_Boosting_Self-organizing_Maps_with_Hyperdimensional_Computing)

## Contributing
Contributions are welcome! Please fork the repository and submit pull requests with any enhancements, bug fixes, or improvements.

## License
Distributed under the MIT License. See `LICENSE` for more information.
