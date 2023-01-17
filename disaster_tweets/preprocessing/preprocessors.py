import re
from typing import Iterable, List, Union

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from keras.utils import pad_sequences
from keras_preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.stem.api import StemmerI
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder


def preprocess_tweet(
    sentence: str,
    remove_stopwords: bool = False,
    stemmer: StemmerI = None,
    tokenize: bool = False,
    to_match: List[str] = None,
) -> Union[str, List[str]]:
    """
    Preprocesses a sentence by removing html tags and converting html encodings, removing punctuation, website addressed, tags, and converting text
    to lower case. Note that the words `'new'` and `'via'` are removed. If `remove_stopwords` then English stopwords will also be removed.
    If `stemmer` is provided then words will also be stemmed. Returns preprocessed sentence as a string if `tokenize=False`, else a list of tokens.

    Parameters
    ----------
    sentence : str
        The sentence to text_preprocess
    remove_stopwords : bool, default=False
        Whether to remove stopwords
    stemmer : nltk.stem.api.StemmerI, optional
        NLTK stemmer, by default None
    tokenize : bool, optional
        Whether to return tokens or string, by default False
    to_match : List[str], optional
        Which characters to keep

    Returns
    -------
    str or List[str]
        If `tokenize` then a list of tokens is returned, otherwise the preprocessed sentence is returned as a string.
    """
    sentence = BeautifulSoup(str(sentence), "lxml").text
    to_match = ["http\S+", "@\S+", "[^a-zA-Z]"] if to_match is None else to_match
    sentence = sentence.lower()
    sentence = re.sub("n't", " not", sentence)
    sentence = re.sub("|".join(to_match), " ", sentence)

    stopword_list = (
        stopwords.words("english") + ["new", "via"]
        if remove_stopwords
        else ["new", "via"]
    )
    tokens = [
        word for word in sentence.split() if word not in stopword_list if len(word) > 1
    ]
    if stemmer is None:
        if not tokenize:
            return " ".join(tokens)
        return tokens
    stemmed = [stemmer.stem(token) for token in tokens]
    if not tokenize:
        return " ".join(stemmed)
    return stemmed


def text_preprocess(
    tweets: Iterable[str],
    stemmer: StemmerI = None,
    tokenize: bool = False,
    to_match: List[str] = None,
    words_to_remove: List[str] = None,
) -> List[Union[str, List[str]]]:
    """
    Preprocesses a collection of tweets using an instance of the `disaster_tweets.preprocessing.TextPreprocessor` class.

    Parameters
    ----------
    tweets : Iterable[str]
        Collection of tweets
    stemmer : StemmerI, optional
        NLTK stemmer, by default None
    tokenize : bool, optional
        Whether to tokenize each sentence, by default False
    to_match : List[str], optional
        Which characters to keep
    words_to_remove : List[str], optional
        List of words to remove from preprocessed tweets.

    Returns
    -------
    List[str] or List[List[str]]
        List containing the preprocessed tweets, either as strings or lists of tokens.
    """
    preprocessor = TextPreprocessor(
        stemmer=stemmer,
        tokenize=tokenize,
        to_match=to_match,
        words_to_remove=words_to_remove,
    )
    return preprocessor.fit_transform(X=tweets)
    # return [sentence if sentence == np.nan else preprocess_tweet(sentence, remove_stopwords, stemmer, tokenize) for sentence in tweets]


class TextPreprocessor:
    """
    sklearn-style preprocessor class for global preprocessing of the `text` column of the disaster tweets dataset. This preprocessor
    is compatible with sklearn pipelines.

    Parameters
    ----------
    stemmer : nltk.stem.api.StemmerI, optional
        NLTK stemmer, by default None
    tokenize : bool, optional
        Whether to return tokens or string, by default False
    to_match : List[str], optional
        Which characters to keep
    words_to_remove : List[str], optional
        List of words to remove from preprocessed tweets.

    Attributes
    ----------
    self.is_fitted : bool
        Whether the `fit` method has been called.
    self.n_samples_in : int
        The number of samples in the data, populated when `fit` is called.
    self.input_shape : Tuple
        The shape of the input data, populated when `fit` is called.
    self.text_vocab_size : int
        The size of the vocabulary in the text data.
    """

    def __init__(
        self,
        stemmer: StemmerI = None,
        tokenize: bool = False,
        to_match: List[str] = None,
        words_to_remove: List[str] = None,
    ):
        self.stemmer = stemmer
        self.tokenize = tokenize
        self.to_match = (
            ["http\S+", "@\S+", "[^a-zA-Z&]"] if to_match is None else to_match
        )
        self.is_fitted = False
        self.words_to_remove = words_to_remove

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y=None):
        """
        Fits the preprocessor. Should be called before `transform`.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input text data.
        y : None
            Not used, only here for compatability, by default None

        Returns
        -------
        Instance of self.
        """
        self.n_samples_in = len(X)
        self.input_shape = X.shape[1:]
        self.is_fitted = True
        return self

    def transform(
        self, X: Union[np.ndarray, pd.DataFrame], y=None
    ) -> Union[List[str], List[List[str]]]:
        """
        Preprocesses the text data in `X`.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input text data.
        y : None
            Not used, only here for compatability, by default None

        Returns
        -------
        List[str] or List[List[str]]
            The preprocessed text, either as strings or list of string tokens according as `self.tokenize`.
        """
        preprocessed_text = [self._transform_sentence(sentence) for sentence in X]
        if not self.tokenize:
            return preprocessed_text
        text_tokenizer = Tokenizer()
        text_tokenizer.fit_on_texts(preprocessed_text)
        text_input = text_tokenizer.texts_to_sequences(preprocessed_text)
        text_input = pad_sequences(text_input)
        self.text_vocab_size = len(text_tokenizer.word_index) + 1
        return text_input

    def fit_transform(
        self, X: Union[np.ndarray, pd.DataFrame], y=None
    ) -> Union[List[str], List[List[str]]]:
        """
        Fits and transforms the data in `X`.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input text data.
        y : None
            Not used, only here for compatability, by default None

        Returns
        -------
        List[str] or List[List[str]]
            The preprocessed text, either as strings or list of string tokens according as `self.tokenize`.
        """
        self.fit(X)
        return self.transform(X)

    def _transform_sentence(self, sentence: str) -> Union[str, List[str]]:
        """
        Helper function that preprocessed a string `sentence` as specified.

        Parameters
        ----------
        sentence : str
            The string to preprocess.

        Returns
        -------
        str or List[str]
            The preprocessed string returned either as a string or a list of tokens according as `self.tokenize`.
        """
        sentence = BeautifulSoup(str(sentence), "lxml").text
        sentence = sentence.lower()
        sentence = re.sub("n't", " not", sentence)
        sentence = re.sub("|".join(self.to_match), " ", sentence)
        tokens = [
            word
            for word in sentence.split()
            if word not in self.words_to_remove
            if len(word) > 1
        ]
        if self.stemmer is None:
            if not self.tokenize:
                return " ".join(tokens)
            return tokens
        stemmed = [self.stemmer.stem(token) for token in tokens]
        if not self.tokenize:
            return " ".join(stemmed)
        return stemmed


class RNNPreprocessor:
    """
    sklearn-style preprocessor class for RNN-specific preprocessing of the feature columns of the disaster tweets dataset. This preprocessor
    is compatible with sklearn pipelines.

    At least one or `text_col`, `meta_col`, `cat_col`, or `num_col` must be specified. These should correspond to the feature inputs to the
    RNN model of `build_RNN` in `disaster_tweets.models.RNN`.

    Parameters
    ----------
    text_col: str, default is None
        Name of the text input column to the RNN. These are tokenized and padded with 0s to have the same length.
    meta_col: str, default is None
        Name of the text metadata input column to the RNN. This column is treated as text input containing metadata on the
        corresponding text from the text column. These are tokenized and padded with 0s to have the same length.
    cat_col: List[str] or str, default is None
        Name of the categorical columns input to the RNN. These are one-hot encoded.
    num_col: str, default is None
        Name of the numerical column input to the RNN.

    Attributes
    ----------
    self.text_input_shape, Tuple[int]
        The shape of the preprocessed text input data, populated when `transform` has been called.
    self.text_vocab_size, int
        The size of the vocabulary in the text input data, populated when `fit` has been called.
    self.text_maxlen, int
        The dimension of the transformed text vectors, equivalent to `self.text_input_shape[1]`.
    self.meta_input_shape, Tuple[int]
        The shape of the preprocessed text metadata, populated when `transform` has been called.
    self.meta_vocab_size, int
        The size of the vocabulary in the text metadata, populated when `fit` has been called.
    self.meta_maxlen, int
        The dimension of the transformed text metadata vectors, equivalent to `self.meta_input_shape[1]`.
    self.cat_input_shape, Tuple[int]
        The shape of the preprocessed categorical data, populated when `tranform` has been called.
    self.num_input_shape, Tuple[int]
        The shape of the preprocessed numerical data, populated when `transform` has been called.
    self.n_samples_in, int
        The number of samples in the input data, populated when `fit` has been called.
    self.is_fitted, bool
        Whether `fit` has been called. `fit` must be called before `transform` is called.
    """

    def __init__(
        self,
        text_col: str = None,
        meta_col: str = None,
        cat_col: Union[List[str], str] = None,
        num_col: str = None,
    ):
        if not any([text_col, meta_col, cat_col, num_col]):
            raise ValueError(
                "At least one of `text_col`, `meta_col`, `cat_col`, and `num_col` must be passed."
            )
        self.text_col = text_col
        self.meta_col = meta_col
        self.cat_col = [cat_col] if isinstance(cat_col, str) else cat_col
        self.num_col = num_col
        self.text_input_shape = None
        self.text_vocab_size = None
        self.text_maxlen = None
        self.meta_input_shape = None
        self.meta_vocab_size = None
        self.meta_maxlen = None
        self.cat_input_shape = None
        self.num_input_shape = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y=None):
        """
        Fits the preprocessor. Should be called before `transform`.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input text data.
        y : None
            Not used, only here for compatability, by default None

        Returns
        -------
        Instance of self.
        """
        self.n_samples_in = len(X)
        if self.text_col is not None:
            self._fit_text(X)
            self.text_vocab_size = len(self.text_tokenizer.word_index) + 1
        if self.meta_col is not None:
            self._fit_meta(X)
            self.meta_vocab_size = len(self.meta_tokenizer.word_index) + 1
        if self.cat_col is not None:
            self._fit_cat(X)
        if self.num_col is not None:
            self._fit_num(X)

        self.is_fitted = True
        return self

    def transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y=None,
        text_maxlen: int = None,
        meta_maxlen: int = None,
    ) -> List[np.ndarray]:
        """
        Preprocesses the text data in `X`. Text and text metadata are returned as tokenized sequences padded with 0s
        to have the same length. Categorical columns are one-hot encoded.

        Use `text_maxlen` and `meta_maxlen` when transforming test data to make sure they have the correct input
        shape to the RNN, e.g., `text_maxlen` for the test data should be set to the maximum length training text vector.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input text data.
        y : None
            Not used, only here for compatability, by default None
        text_maxlen: int, optional
            Sets the length of the transformed text data. By default, this is set to the longest tokenized sentence
            and is not required for the training data. When transforming test data, this should be set to the
            maximum-length vector of the text data in the original training data.
        meta_maxlen: int, optional
            Sets the length of the transformed text metadata. By default, this is set to the longest tokenized sentence
            and is not required for the training data. When transforming test data, this should be set to the
            maximum-length vector of the metadata in the original training data.

        Returns
        -------
        List[np.ndarray]
            A list of the transformed data. The length of the list will be equal to the number of columns input
            to the constructor and will be a maximum of 4.
            Input data will appear in the following order `[text, meta, cat, num]`.
        """

        if not self.is_fitted:
            raise NotFittedError(
                "`RNNPreprocessor` not fitted, call `fit` method before calling `transform`."
            )

        inputs = []
        if self.text_col is not None:
            text_input = X[self.text_col].fillna(value="<UNK>").values.tolist()
            inputs.append(self._transform_text(text_input, maxlen=text_maxlen))
        if self.meta_col is not None:
            meta_input = X[self.meta_col].fillna(value="<UNK>").values.tolist()
            inputs.append(self._transform_meta(meta_input, maxlen=meta_maxlen))
        if self.cat_col is not None:
            cat_input = X[self.cat_col]
            inputs.append(self._transform_cat(cat_input))
        if self.num_col is not None:
            num_input = X[self.num_col]
            inputs.append(self._transform_num(num_input))
        return inputs

    def fit_transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y=None,
        text_maxlen: int = None,
        meta_maxlen: int = None,
    ) -> List[np.ndarray]:
        """
        Fits and transforms the data in `X` by calling `fit` and `transform` methods together.

        Use `text_maxlen` and `meta_maxlen` when transforming test data to make sure they have the correct input
        shape to the RNN, e.g., `text_maxlen` for the test data should be set to the maximum length training text vector.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input text data.
        y : None
            Not used, only here for compatability, by default None
        text_maxlen: int, optional
            Sets the length of the transformed text data. By default, this is set to the longest tokenized sentence
            and is not required for the training data. When transforming test data, this should be set to the
            maximum-length vector of the text data in the original training data.
        meta_maxlen: int, optional
            Sets the length of the transformed text metadata. By default, this is set to the longest tokenized sentence
            and is not required for the training data. When transforming test data, this should be set to the
            maximum-length vector of the metadata in the original training data.

        Returns
        -------
        List[np.ndarray]
            A list of the transformed data. The length of the list will be equal to the number of columns input
            to the constructor and will be a maximum of 4.
            Input data will appear in the following order `[text, meta, cat, num]`.
        """
        self.fit(X, y)
        return self.transform(X, y, text_maxlen=text_maxlen, meta_maxlen=meta_maxlen)

    def _fit_text(self, X: Union[np.ndarray, pd.DataFrame]):
        """
        Helper function that fits the text col

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input text data

        Returns
        -------
        Instance of self
        """
        text_input = X[self.text_col].fillna(value="<UNK>").values.tolist()
        self.text_tokenizer = Tokenizer()
        self.text_tokenizer.fit_on_texts(text_input)
        return self

    def _fit_meta(self, X: Union[np.ndarray, pd.DataFrame]):
        """
        Helper function that fits the text metadata col

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input text data

        Returns
        -------
        Instance of self
        """
        meta_input = X[self.meta_col].fillna(value="<UNK>").values.tolist()
        self.meta_tokenizer = Tokenizer()
        self.meta_tokenizer.fit_on_texts(meta_input)
        return self

    def _fit_cat(self, X: Union[np.ndarray, pd.DataFrame]):
        """
        Helper function that fits the categorical col

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input text data

        Returns
        -------
        Instance of self
        """
        cat_input = X[self.cat_col]
        self.one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        self.one_hot_encoder.fit(cat_input)
        return self

    def _fit_num(self, X: Union[np.ndarray, pd.DataFrame]):
        """
        Helper function that fits the numerical col

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input text data

        Returns
        -------
        Instance of self
        """
        raise NotImplementedError

    def _transform_text(self, text: List[str], maxlen: int = None) -> np.ndarray:
        """
        Helper function that transforms the text data by tokenizing and pad sequences with 0 to have the same length.

        By default sequences are padded to have the length of the longest tokenized sentence. Set `maxlen` to change this.

        Parameters
        ----------
        text : List[str]
            Text data to transform
        maxlen : int, optional
            Length of the transformed tokenized text, which are padded with 0s to have the same length,
            by default this will be set to the longest length tokenized sentence in the tokenized data.

        Returns
        -------
        np.ndarray
            Tokenized and padded text.
        """
        tokenized = self.text_tokenizer.texts_to_sequences(text)
        self.text_maxlen = max(list(map(len, tokenized)))
        text_input = pad_sequences(tokenized, maxlen=maxlen)
        self.text_input_shape = text_input.shape[1:]
        return text_input

    def _transform_meta(self, meta: List[str], maxlen: int = None) -> np.ndarray:
        """
        Helper function that transforms the text metadata by tokenizing and pad sequences with 0 to have the same length.

        By default sequences are padded to have the length of the longest tokenized sentence. Set `maxlen` to change this.

        Parameters
        ----------
        meta : List[str]
            Text metadata to transform
        maxlen : int, optional
            Length of the transformed tokenized text, which are padded with 0s to have the same length,
            by default this will be set to the longest length tokenized sentence in the tokenized data.

        Returns
        -------
        np.ndarray
            Tokenized and padded text metadata.
        """
        tokenized = self.meta_tokenizer.texts_to_sequences(meta)
        self.meta_maxlen = max(list(map(len, tokenized)))
        meta_input = pad_sequences(tokenized, maxlen=maxlen)
        self.meta_input_shape = meta_input.shape[1:]
        return meta_input

    def _transform_cat(self, cat: pd.DataFrame) -> np.ndarray:
        """
        Helper function that transforms categorical features by one-hot encoding.

        Parameters
        ----------
        cat : pd.Dataframe
            DataFrame containing only categorical features

        Returns
        -------
        np.ndarray
            One-hot encoded categorical features
        """
        cat_input = self.one_hot_encoder.transform(cat)
        self.cat_input_shape = cat_input.shape[1:]
        return cat_input

    def _transform_num(self, num):
        """
        Helper function that transforms the numerical column.

        Parameters
        ----------
        num : pd.DataFrame
            DataFrame containing the numerical column.

        Returns
        -------

        """
        return num
