from typing import Tuple

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def naive_bayes_pipeline(
    ngram_range: Tuple[int] = (1, 1), verbose: bool = True
) -> Pipeline:
    """
    `sklearn` pipeline for multinomial Naive Bayes model consisting of a count vectorizer preprocessing step and
    multinomial Naive Bayes model step.

    Parameters
    ----------
    ngram_range : Tuple[int], optional
        The ngram range to use, this will use ngrams for all n between ngram_range[0] and ngram_range[1] inclusive.
        By default (1, 1).
    verbose : bool, optional
        Whether to print training information, by default True

    Returns
    -------
    sklearn.pipeline.Pipeline
        sklearn pipeline with count vectorizer and naive bayes step.
    """
    steps = [
        ("vectorizer", CountVectorizer(ngram_range=ngram_range)),
        ("naive_bayes", MultinomialNB()),
    ]
    return Pipeline(steps=steps, verbose=verbose)
