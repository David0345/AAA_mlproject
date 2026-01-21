from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import FastText
from tqdm.auto import tqdm
import numpy as np
import re


class TextPreprocessor:
    """
    Text preprocessing pipeline using FastText embeddings with TF-IDF weighting.

    converts text data (title and description) into vector
    by combining FastText word embeddings with TF-IDF

    - Text tokenization and stemming
    - FastText model training (300-dim)
    - TF-IDF vectorization
    - averaging BOE + TF-IDF

    Attributes
    ----------
    tfidfvectorizer : TfidfVectorizer or None
        Trained TF-IDF vectorizer for computing word importance weights
    fasttext_model : FastText or None
        Trained FastText model for generating word embeddings

    Methods
    -------
    fit_transform(X: pd.DataFrame) -> np.ndarray
        Trains FastText and TF-IDF models on the input data and returns embeddings

    transform(X: pd.DataFrame) -> np.ndarray
        Applies trained models to transform new data into embeddings
    """
    def __init__(self):
        self.tfidfvectorizer = None
        self.fasttext_model = None
    def fit_transform(self, X):
        X = get_enhanced_texts(X, True)

        tfidf_vectorizer = TfidfVectorizer(
            analyzer=lambda x: x,
            lowercase=False,
            token_pattern=None,
            min_df=5,
            max_df=0.1
        )

        self.tfidfvectorizer = tfidf_vectorizer

        self.tfidfvectorizer.fit(X)
        ft = FastText(
            vector_size=300,
            window=5,
            min_count=5,
            workers=8
        )

        self.fasttext_model = ft
        self.fasttext_model.build_vocab(corpus_iterable=X)
        self.fasttext_model.train(
            corpus_iterable=X,
            total_examples=len(X),
            epochs=10
        )

        fasttext_embeddings = get_fasttext_embeddings(
            tokenized_texts=X,
            model=self.fasttext_model,
            tfidf_vectorizer=self.tfidfvectorizer
        )

        return fasttext_embeddings

    def transform(self, X):
        if self.tfidfvectorizer is None or self.fasttext_model is None:
            raise ValueError('You need to fit the model before transforming')

        X = get_enhanced_texts(X, True)

        fasttext_embeddings = get_fasttext_embeddings(
            tokenized_texts=X,
            model=self.fasttext_model,
            tfidf_vectorizer=self.tfidfvectorizer
        )

        return fasttext_embeddings


def get_enhanced_texts(data, test=False, keep_stopwords=False):
    """Tokenize and stem text data with optional stopword removal.

    Processes title and description columns by:
    1. Combining title and description
    2. Converting to lowercase
    3. Tokenizing with NLTK word_tokenize
    4. Removing non-word characters
    5. Stemming with Snowball stemmer
    6. Filtering stopwords (optional)

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe with 'title' and 'description' columns.
        If test=False, must also contain target columns:
        ['real_weight', 'real_height', 'real_length', 'real_width']
    test : bool, default=False
        If True, returns only tokenized texts.
        If False, returns tokenized texts and target values.
    keep_stopwords : bool, default=False
        If True, keeps Russian stopwords in the text.
        If False, removes common Russian stopwords.

    Returns
    -------
    List[List[str]] or Tuple[List[List[str]], np.ndarray]
        If test=True:
            List of tokenized and stemmed documents
        If test=False:
            Tuple of (tokenized_texts, target_values)
            where target_values is array of shape (n_samples, 4)
    """
    titles = data.title.astype(str).fillna('').values
    descriptions = data.description.astype(str).fillna('').values

    # stop words
    minimal_stopwords = set() if keep_stopwords else {
        'и', 'в', 'на', 'с', 'по', 'из', 'к', 'у', 'о', 'об', 'от', 'до',
        'для', 'при', 'за', 'под', 'над', 'то', 'вы', 'мы', 'они', 'он',
        'она', 'это', 'всё', 'так', 'же', 'бы', 'был', 'была', 'были'
    }

    stemmer = SnowballStemmer('russian')
    tokenized_texts = []

    non_word_pattern = re.compile(r'[^\w]', flags=re.UNICODE)

    print(f'Processing {len(titles)} texts...')

    for idx, (title, desc) in enumerate(zip(titles, descriptions)):
        if (idx + 1) % 10000 == 0:
            print(f'  Processed {idx + 1}/{len(titles)}')

        # lowering
        full_text = f'{title}. {desc}'.lower()

        # tokenization
        tokens = word_tokenize(full_text)

        tokens = [non_word_pattern.sub('', token) for token in tokens]

        tokens = [
            stemmer.stem(token) for token in tokens
            if token and token not in minimal_stopwords
        ]

        tokenized_texts.append(tokens)

    if not test:
        targets = data[['real_weight', 'real_height', 'real_length', 'real_width']].values
        return tokenized_texts, targets

    return tokenized_texts

def get_word_vector(model, word):
    if hasattr(model, 'wv'):
        return model.wv[word]
    else:
        return model[word]


def get_fasttext_embeddings(tokenized_texts, model, tfidf_vectorizer=None):
    """
    extract word vector from FastText model.

    Parameters
    ----------
    model : FastText
        Trained FastText model or KeyedVectors object
    word : str
        Word to get the vector for

    Returns
    -------
    np.ndarray
        Word vector of shape (vector_size,)
    """
    embeddings = []

    # gettinc word2index dict
    word2index = None
    if tfidf_vectorizer:
        feature_names = tfidf_vectorizer.get_feature_names_out()
        word2index = {word: idx for idx, word in enumerate(feature_names)}

    iterator = tqdm(tokenized_texts, desc='Генерация эмбеддингов')

    for tokens in iterator:
        if len(tokens) == 0:
            embeddings.append(np.zeros(300))
            continue

        token_vectors = []
        weights = []

        # tf-idf weights
        current_tfidf_weights = {}
        if tfidf_vectorizer:
            try:
                text_str = ' '.join(map(str, tokens))
                response = tfidf_vectorizer.transform([text_str])
                row_indices, col_indices = response.nonzero()
                for col in col_indices:
                    current_tfidf_weights[col] = response[0, col]
            except Exception:
                pass

        for token in tokens:
            token_str = str(token)

            try:
                vector = get_word_vector(model, token_str)
                token_vectors.append(vector)

                weight = 1.0
                if tfidf_vectorizer and word2index:
                    idx = word2index.get(token_str)
                    if idx is not None:
                        weight = current_tfidf_weights.get(idx, 1.0)

                weights.append(weight)

            except KeyError:
                continue

        if token_vectors:
            if weights:
                weights = np.array(weights)
                if np.sum(weights) == 0:
                    weights = None

            embedding = np.average(token_vectors, axis=0, weights=weights)
        else:
            embedding = np.zeros(300)

        embeddings.append(embedding)

    return np.array(embeddings)