from disaster_tweets.pipelines import naive_bayes_pipeline, logistic_regression_pipeline

# directories ----------------------------------
DIR_PATH = "/Users/salvatoremercuri/python-projects/disaster-tweets/"
TRAIN_PATH = "data/train.csv"
PREPROC_TRAIN_PATH = "data/preproc_train.csv"
TEST_PATH = "data/test.csv"
PREPROC_TEST_PATH = "data/preproc_test.csv"
OUTPUT_DIR = "output"
PREDS_PATH = "output/test_preds.csv"
MODEL_PATH = "output/rnn"

# modelling ------------------------------------
TEXT_COL = "text"
META_COL = None
CAT_COL = "keyword"
NUM_COL = None
FEATURE_COLS = [
    col for col in [TEXT_COL, META_COL, CAT_COL, NUM_COL] if col is not None
]
LABEL_COL = "target"
VAL_SPLIT = 0.2
NGRAM_RANGE = (1, 3)
NB_PARAM_GRID = {"vectorizer__ngram_range": [(1, i) for i in range(1, 11)]}
RNN_PARAM_GRID = {
    "epochs": [5, 10, 20],
    "batch_size": [16, 32, 64, 128],
    "optimizer": ["SGD", "Adam", "Adagrad"],
    # "optimizer__learning_rate": [0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
    # "optimizer__momentum": [0.0, 0.2, 0.4, 0.6, 0.8],
    "activation": ["relu", "tanh", "sigmoid"],
    "units": [128, 256, 512, 1024],
    "embedding_output_dim": [64, 128, 256, 512],
}

TEXT_EMBEDDING_DIM = 128
META_EMBEDDING_DIM = 32

# other ----------------------------------------
SEED = 1729
EXPORT_TEST_PREDS = True
