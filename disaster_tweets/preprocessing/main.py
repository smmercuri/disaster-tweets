import warnings

import pandas as pd

from disaster_tweets.config import (
    DIR_PATH,
    PREPROC_TEST_PATH,
    PREPROC_TRAIN_PATH,
    TEST_PATH,
    TRAIN_PATH,
)
from disaster_tweets.preprocessing import text_preprocess

warnings.filterwarnings("ignore")

print("\nLoading data...")
df = pd.read_csv(DIR_PATH + TRAIN_PATH)
df_test = pd.read_csv(DIR_PATH + TEST_PATH)

print(f"\nPreprocessing data...")
df = df.drop_duplicates(subset=["text"], keep=False)
df["text"] = text_preprocess(df["text"], words_to_remove=["new", "via"])
df_test["text"] = text_preprocess(df_test["text"], words_to_remove=["new", "via"])
df["keyword"] = text_preprocess(df["keyword"])
df_test["keyword"] = text_preprocess(df_test["keyword"])
print(f"Data preprocessed.\n")

df.to_csv(DIR_PATH + PREPROC_TRAIN_PATH)
df_test.to_csv(DIR_PATH + PREPROC_TEST_PATH)
