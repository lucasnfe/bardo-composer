import json
import argparse

import numpy as np
import tensorflow as tf
import transformers as tm

from models import *
from data_vgmidi import *

def build_clf_model(vocab_size, params):
    # Create GPT2 languade model configuration
    clf_config = tm.GPT2Config(vocab_size, params["seqlen"], params["n_ctx"], params["embed"], params["layers"], params["heads"],
                               resid_pdrop=params["drop"], embd_pdrop=params["drop"], attn_pdrop=params["drop"])

    # Load pre-trained GPT2 without language model head
    clf_gpt2 = GPT2Classifier(clf_config)
    if params["finetune"]:
        ckpt = tf.train.Checkpoint(net=clf_gpt2)
        ckpt.restore(tf.train.latest_checkpoint(params["pretr"]))

    return clf_gpt2

def train_clf_model(model, params, train_dataset, test_dataset):
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(params["lr"]), metrics=['accuracy'])

    checkpoint = tf.keras.callbacks.ModelCheckpoint('../../trained/clf_gpt2.ckpt/clf_gpt2' + "_" + str(params["dimension"]) + "/clf_gpt2",
        monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True)

    history = model.fit(train_dataset, epochs=params["epochs"], validation_data=test_dataset, callbacks=[checkpoint])

    return history

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
    train_text = load_dataset(params["train"], vocab, params["seqlen"], params["dimension"])
    test_text = load_dataset(params["test"], vocab, params["seqlen"], params["dimension"])

    train_dataset = build_dataset(train_text, params["batch"])
    test_dataset = build_dataset(test_text, params["batch"])

    # Calculate vocab_size from char2idx dict
    vocab_size = len(vocab)

    # Build gpt2
    clf_gpt2 = build_clf_model(vocab_size, params)

    # Train gpt2
    train_clf_model(clf_gpt2, params, train_dataset, test_dataset)
