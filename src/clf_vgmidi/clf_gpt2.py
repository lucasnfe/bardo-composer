import os
import csv
import json
import argparse

import numpy as np
import tensorflow as tf
import transformers as tm
import midi.encoder as me

BUFFER_SIZE=10000

from models import *

def load_dataset(datapath, vocab, seq_length):
    dataset = []

    data = csv.DictReader(open(datapath, "r"))
    for row in data:
        filepath, label = row["filepath"], int(row["label"])

        piece_path = os.path.join(os.path.dirname(datapath), filepath)
        piece_text = me.load_file(piece_path).split(" ")
        tokens = [vocab[c] for c in piece_text]

        dataset.append((tokens[:seq_length], [label]))

    return dataset

def build_dataset(dataset, batch_size):
    # Read all files in the dataset directory and combine them
    tf_dataset = tf.data.Dataset.from_generator(lambda: dataset, (tf.int32, tf.int32))
    tf_dataset = tf_dataset.shuffle(BUFFER_SIZE)
    tf_dataset = tf_dataset.padded_batch(batch_size, padded_shapes=([None], [1]), padding_values=(1, 1))

    return tf_dataset

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description='clf_gp2.py')
    parser.add_argument('--conf', type=str, required=True, help="JSON file with training parameters.")
    opt = parser.parse_args()

    # Load training parameters
    params = {}
    with open(opt.conf) as conf_file:
        params = json.load(conf_file)["clf_gpt2"]

    # Load char2idx dict from json file
    with open(params["vocab"]) as f:
        vocab = json.load(f)

    # Build dataset from encoded unlabelled midis
    train_text = load_dataset(params["train"], vocab, params["seqlen"])
    test_text = load_dataset(params["test"], vocab, params["seqlen"])

    train_dataset = build_dataset(train_text, params["batch"])
    test_dataset = build_dataset(test_text, params["batch"])

    # Calculate vocab_size from char2idx dict
    vocab_size = len(vocab)

    # Create GPT2 languade model configuration
    clf_config = tm.GPT2Config(vocab_size, params["seqlen"], params["n_ctx"], params["embed"], params["layers"], params["heads"],
                               resid_pdrop=params["drop"], embd_pdrop=params["drop"], attn_pdrop=params["drop"])

    # Load pre-trained GPT2 without language model head
    clf_gpt2 = GPT2Classifier(clf_config)
    if params["finetune"]:
        ckpt = tf.train.Checkpoint(net=clf_gpt2)
        ckpt.restore(tf.train.latest_checkpoint(params["pretr"]))

    clf_gpt2.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(params["lr"]), metrics=['accuracy'])

    checkpoint = tf.keras.callbacks.ModelCheckpoint('../../trained/clf_gpt2.ckpt/clf_gpt2',
        monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True)

    history = clf_gpt2.fit(train_dataset, epochs=params["epochs"], validation_data=test_dataset, callbacks=[checkpoint])
