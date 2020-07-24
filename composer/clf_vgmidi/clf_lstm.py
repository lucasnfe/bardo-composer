import json
import argparse

import numpy as np
import tensorflow as tf
import transformers as tm

from models import *
from data_vgmidi import *

from sklearn.linear_model import LogisticRegression

# Directory where trained model will be saved
TRAIN_DIR = "./trained"

def build_clf_model(vocab_size, params):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Embedding(vocab_size, embed_dim, batch_input_shape=[batch_size, None]))

    for i in range(max(1, lstm_layers)):
        model.add(tf.keras.layers.LSTM(lstm_units, return_sequences=True, stateful=True, dropout=dropout, recurrent_dropout=dropout))

    model.add(tf.keras.layers.Dense(vocab_size))

    return model

def train_clf_classifier(clf_lstm, params, train_dataset, test_dataset):
    clf_lstm.compile(loss=tf.keras.losses.sparse_categorical_crossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(params["lr"]), metrics=['accuracy'])

    checkpoint = tf.keras.callbacks.ModelCheckpoint('../../trained/clf_lstm.ckpt/clf_lstm' + "_" + str(params["dimension"]) + "/clf_lstm",
        monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True)

    history = clf_lstm.fit(train_dataset, epochs=params["epochs"], validation_data=test_dataset, callbacks=[checkpoint])

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='clf_lstm.py')
    parser.add_argument('--conf', type=str, required=True, help="JSON file with training parameters.")
    opt = parser.parse_args()

    # Load training parameters
    params = {}
    with open(opt.conf) as conf_file:
        params = json.load(conf_file)["clf_lstm"]

    # Load char2idx dict from json file
    with open(params["vocab"]) as f:
        vocab = json.load(f)

    # Build dataset from encoded unlabelled midis
    train_text = load_dataset(params["train"], vocab, params["seqlen"], params["dimension"])
    test_text = load_dataset(params["test"], vocab, params["seqlen"], params["dimension"])

    train_dataset = build_dataset(train_text, params["batch"])
    test_dataset = build_dataset(test_text, params["batch"])

    # Calculate vocab_size from char2idx dict
    vocab_size = len(vocab)

    # Rebuild generative model from checkpoint
    clf_lstm = build_clf_model(vocab_size, params)
