import os
import warnings

import pandas as pd
from keras.optimizers import SGD, Adagrad, Adam
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from disaster_tweets.config import (
    CAT_COL,
    EXPORT_TEST_PREDS,
    FEATURE_COLS,
    LABEL_COL,
    META_COL,
    META_EMBEDDING_DIM,
    MODEL_PATH,
    NB_PARAM_GRID,
    NGRAM_RANGE,
    NUM_COL,
    PREDS_PATH,
    PREPROC_TEST_PATH,
    PREPROC_TRAIN_PATH,
    RNN_PARAM_GRID,
    TEXT_COL,
    TEXT_EMBEDDING_DIM,
    VAL_SPLIT,
)
from disaster_tweets.models import build_RNN
from disaster_tweets.pipelines import logistic_regression_pipeline, naive_bayes_pipeline
from disaster_tweets.preprocessing import RNNPreprocessor

warnings.filterwarnings("ignore")

# Data ----------------------------------------------------------
print("\nLoading preprocessed data...")
if not os.path.exists(PREPROC_TRAIN_PATH) or not os.path.exists(
    PREPROC_TEST_PATH
):
    raise FileNotFoundError(
        "Preprocessed data not found. Run `disaster_tweets.preprocessing.main.py` first to generate the preprocessed data."
    )
df = pd.read_csv(PREPROC_TRAIN_PATH)
df_test = pd.read_csv(PREPROC_TEST_PATH)
print("Data loaded.")

X_train, y_train = df[FEATURE_COLS], df[LABEL_COL]
X_test = df_test[FEATURE_COLS]
print(f"Training input shape: {X_train.shape}.\n")

preprocessor = RNNPreprocessor(
    text_col=TEXT_COL, meta_col=META_COL, cat_col=CAT_COL, num_col=NUM_COL
)
inputs = preprocessor.fit_transform(X_train)
if EXPORT_TEST_PREDS:
    test_inputs = preprocessor.transform(
        X_test,
        text_maxlen=preprocessor.text_maxlen,
        meta_maxlen=preprocessor.meta_maxlen,
    )

learning_rate = 0.001
model = build_RNN(
    text_input_shape=preprocessor.text_input_shape,
    embedding_input_dim=preprocessor.text_vocab_size,
    embedding_output_dim=TEXT_EMBEDDING_DIM,
    meta_input_shape=preprocessor.meta_input_shape,
    meta_embedding_input_dim=preprocessor.meta_vocab_size,
    meta_embedding_output_dim=META_EMBEDDING_DIM,
    cat_input_shape=preprocessor.cat_input_shape,
    num_input_shape=preprocessor.num_input_shape,
    units=512,
    activation="tanh",
    dropout=0.2,
    optimizer=Adam(learning_rate=learning_rate),
)
model.fit(
    inputs, y_train, epochs=20, batch_size=32, validation_split=VAL_SPLIT, workers=-1
)
model.save(MODEL_PATH)

# Cross-validation ------
# param_grid = {"vectorizer__ngram_range": [(1, i) for i in range(1, 5)]}
# clf_cv = GridSearchCV(ESTIMATOR, PARAM_GRID, verbose=1)
# rnn_cv = RandomizedSearchCV(model, RNN_PARAM_GRID, verbose=2)
# print(f"\nPerforming cross validation...")

# rnn_cv.fit(inputs[0], y_train)
# print("\nBest parameters: {rnn_cv.best_params_}.\n")
# model = rnn_cv.best_estimator_

# clf_cv.fit(X_train, y_train)

# print(f"\nBest parameters: {clf_cv.best_params_}.\n")

# model = clf_cv.best_estimator_

# Results ---------------
if EXPORT_TEST_PREDS:
    preds = model.predict(test_inputs)
    preds_df = pd.DataFrame(
        {"id": df_test.id, "target": (preds.reshape(-1) > 0.5).astype(int)}
    )
    preds_df.to_csv(PREDS_PATH, index=False)
