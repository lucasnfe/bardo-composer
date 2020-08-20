import json
import argparse

import numpy as np
import tensorflow as tf
import transformers as tm

from load_data import *

def build_clf_model(vocab_size, params):
    # Build Transformer for Language Modeling task
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Embedding(vocab_size, params["embed"], batch_input_shape=[params["batch"], None]))

    for i in range(max(1, params["layers"])):
        model.add(tf.keras.layers.LSTM(params["hidden"], return_sequences=True, stateful=True, dropout=params["drop"]))

    model.add(tf.keras.layers.Dense(vocab_size))

    return model

def train_clf_model(model, params, train_dataset, test_dataset):
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
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
    train_dataset = load_dataset(params["train"], vocab, params["dimension"])
    test_dataset = load_dataset(params["test"], vocab, params["dimension"])

    train_dataset = build_dataset(train_dataset, params["batch"])
    test_dataset = build_dataset(test_dataset, params["batch"])

    # Calculate vocab_size from char2idx dict
    vocab_size = len(vocab)

    # Rebuild generative model from checkpoint
    clf_lstm = build_clf_model(vocab_size, params)
    if params["finetune"]:
        ckpt = tf.train.Checkpoint(net=clf_lstm)
        ckpt.restore(tf.train.latest_checkpoint(params["pretr"]))

    # Add emotion head
    clf_lstm.add(tf.keras.layers.Dense(1, name="emotion_head"))

    # Train lstm
    train_clf_model(clf_lstm, params, train_dataset, test_dataset)
