from typing import Tuple

import keras
from keras.layers import (
    GRU,
    LSTM,
    Activation,
    BatchNormalization,
    Bidirectional,
    Concatenate,
    Dense,
    Dropout,
    Embedding,
    Input,
    SpatialDropout1D,
)
from keras.models import Model, Sequential
from keras.optimizers import Adam


def build_RNN(
    units: int,
    activation: str,
    text_input_shape: Tuple[int] = None,
    embedding_input_dim: int = None,
    embedding_output_dim: int = None,
    meta_input_shape: Tuple[int] = None,
    meta_embedding_input_dim: int = None,
    meta_embedding_output_dim: int = None,
    cat_input_shape: Tuple[int] = None,
    num_input_shape: Tuple[int] = None,
    dropout: float = 0.2,
    optimizer: keras.optimizers.Optimizer = Adam(),
) -> keras.engine.functional.Functional:
    """
    Build function for a deep Keras RNN that consists of up to four parallel architectural sections:
    1. Text embedding and bidirectional RNN for the primary text of the tweets.
    2. Metadata embedding and bidirectional RNN for any separate metadata columns to be treated as text. For example
       the `keyword` or `location` columns may be chosen to be treated as text metadata.
    3. Categorical input layer. This is a single input layer to be used for one-hot encoded categorical data. For example,
       the `keyword` or `location` columns may be chosen to be treated as categorical data.
    4. Numerical input layer. This is a single input layer to be used for numerical data. For example, `location` that has
       been preprocessed to co-ordinates may be fed into this layer.

    Whether to use sections 1.--4. is determined by whether `text_input_shape` (1), `meta_input_shape` (2), `cat_input_shape` (3),
    `num_input_shape` are not None. Therefore at least one of these parameters must be not none. If more than one section is
    used, then the outputs are concatenated together. This is then fed into stacked dense layers before final output
    through a sigmoid activation layer.

    Text data is assumed to be preprocessed into padded sequences of tokens. Categorical data is assumed to be one-hot encoded.
    Use `RNNPreprocessor` from `disaster_tweets.preprocessing.preprocessors` to preprocess the data accordingly.

    Parameters
    ----------
    units : int
        The number of units to use, i.e., the width of the model. Stacked layers will sequentially use half the
        units of the previous layer.
    activation : str
        Keras string identifier for the activation function, e.g., 'tanh', 'sigmoid', 'relu'.
    text_input_shape : Tuple[int], optional
        The text input shape. If using `RNNPreprocessor` this can be accessed through the `text_input_shape` attribute.
        By default None, in which case a text embedding and RNN layer will not form part of the model.
    embedding_input_dim : int, optional
        The input dimension of the text RNN section. This is the dimensional space of the tokens in the preprocessed
        text data. If using `RNNPreprocessor` this can be accessed through the `text_vocab_size` attribute.
        By default None.
    embedding_output_dim : int, optional
        The vector dimension of the text embedding in the text RNN section, by default None.
    meta_input_shape : Tuple[int], optional
        The text metadata input shape. If using `RNNPreprocessor` this can be accessed through the `meta_input_shape` attribute.
        By default None, in which case a meta embedding and RNN layer will not form part of the model.
    meta_embedding_input_dim : int, optional
        The input dimension of the metadata text RNN section. This is the dimensional space of the tokens in the preprocessed
        text metadata. If using `RNNPreprocessor` this can be accessed through the `meta_vocab_size` attribute.
        By default None.
    meta_embedding_output_dim : int, optional
        The vector dimension of the text metadata embedding in the metadata RNN section, by default None.
    cat_input_shape : Tuple[int], optional
        The input shape of the categorical data. If using `RNNPreprocessor` this can be accessed through the `cat_input_shape`
        attribute. By default None, in which categorical input layer and concatenation will not occur.
    num_input_shape : Tuple[int], optional
        The input shape of the numerical data. If using `RNNPreprocessor` this can be accessed through the `num_input_shape`
        attribute. By default None, in which numerical input layer and concatenation will not occur.
    dropout : float, optional
        The dropout rate, which is applied spatially after embedding layers, reccurently in the recurrent layers, and after
        each densely connected layer, by default 0.2.
    optimizer : keras.optimizers.Optimizer, optional
        The keras optimizer to use, by default keras.optimizer.Adam()

    Returns
    -------
    keras.engine.functional.Functional
        Compiled keras model using the functional API.

    Raises
    ------
    ValueError
        If all of the `text`, `meta`, `cat`, or `num` parameters are None. At least one must be used.
    """

    use_text = (
        text_input_shape is not None
        and embedding_input_dim is not None
        and embedding_output_dim is not None
    )
    use_meta = (
        meta_input_shape is not None
        and meta_embedding_input_dim is not None
        and meta_embedding_output_dim is not None
    )
    use_cat = cat_input_shape is not None
    use_num = num_input_shape is not None

    if not any([use_text, use_meta, use_cat, use_num]):
        raise ValueError(
            "At least one of `text`, `meta`, `cat`, `num` input parameters must be supplied."
        )

    inputs = []
    # text embedding and recurrent layers
    if use_text:
        text_input = Input(shape=text_input_shape)
        embedding = Embedding(
            input_dim=embedding_input_dim, output_dim=embedding_output_dim
        )(text_input)
        sp_dp = SpatialDropout1D(dropout)(embedding)
        bn = BatchNormalization()(sp_dp)
        lstm = Bidirectional(
            LSTM(
                units, dropout=dropout, recurrent_dropout=dropout, activation=activation
            )
        )(bn)
        inputs.append(text_input)

    # meta data layers
    if use_meta:
        meta_input = Input(shape=meta_input_shape)
        meta_embedding = Embedding(
            input_dim=meta_embedding_input_dim, output_dim=meta_embedding_output_dim
        )(meta_input)
        meta_sp_dp = SpatialDropout1D(dropout)(meta_embedding)
        meta_bn = BatchNormalization()(meta_sp_dp)
        meta_lstm = Bidirectional(
            GRU(
                units, dropout=dropout, recurrent_dropout=dropout, activation=activation
            )
        )(meta_bn)
        inputs.append(meta_input)

    # categorical input
    if use_cat:
        cat_input = Input(shape=cat_input_shape)
        inputs.append(cat_input)

    # deal with possible concatenations
    # TODO: simplify
    if use_text and inputs == [text_input]:
        dense_input = lstm
    if use_meta and inputs == [meta_input]:
        dense_input = meta_lstm
    if use_cat and inputs == [cat_input]:
        dense_input = cat_input
    if use_text and use_meta and inputs == [text_input, meta_input]:
        dense_input = Concatenate()([lstm, meta_lstm])
    if use_text and use_cat and inputs == [text_input, cat_input]:
        dense_input = Concatenate()([lstm, cat_input])
    if use_meta and use_cat and inputs == [meta_input, cat_input]:
        dense_input = Concatenate()([meta_lstm, cat_input])
    if (
        use_text
        and use_cat
        and use_meta
        and inputs == [text_input, meta_input, cat_input]
    ):
        dense_input = Concatenate()([lstm, meta_lstm, cat_input])

    # dense outputs
    dense_2 = Dense(units)(dense_input)
    bn_2 = BatchNormalization()(dense_2)
    act_2 = Activation(activation)(bn_2)
    dp_2 = Dropout(dropout)(act_2)
    dense_3 = Dense(units // 2)(dp_2)
    bn_3 = BatchNormalization()(dense_3)
    act_3 = Activation(activation)(bn_3)
    dp_3 = Dropout(dropout)(act_3)
    dense_4 = Dense(units // 4)(dp_3)
    bn_4 = BatchNormalization()(dense_4)
    act_4 = Activation(activation)(bn_4)
    dp_4 = Dropout(dropout)(act_4)
    dense_4 = Dense(1)(dp_4)
    outputs = Activation("sigmoid")(dense_4)

    model = Model(inputs=inputs, outputs=[outputs])
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    return model
