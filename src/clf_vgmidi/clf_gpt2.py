import os
import csv
import json
import argparse

import numpy as np
import tensorflow as tf
import transformers as tm
import midi.encoder as me

from gpt2.models import *

def _split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

def load_dataset(datapath, vocab, seq_length):
    dataset = []

    data = csv.DictReader(open(datapath, "r"))
    for row in data:
        filepath, label = row["filepath"], row["label"]

        piece_path = os.path.join(os.path.dirname(datapath), filepath)
        piece_text = me.load_file(piece_path).split(" ")
        tokens = [vocab[c] for c in piece_text]

        if len(tokens) < seq_length - 1:
            input_text, target_text = _split_input_target(tokens)
            dataset.append((input_text, target_text, [label]))

    return dataset

def _split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

def build_dataset(dataset, batch_size, BUFFER_SIZE=10000):
    # Read all files in the dataset directory and combine them
    tf_dataset = tf.data.Dataset.from_generator(lambda: dataset, (tf.int32, tf.int32, tf.int32))
    tf_dataset = tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    tf_dataset = tf_dataset.shuffle(BUFFER_SIZE)
    tf_dataset = tf_dataset.padded_batch(batch_size, padded_shapes=([None],[None], [1]), padding_values=(1, 1, 1))

    return tf_dataset

def loss(y_hat1, y_hat2, y1, y2, alpha=0.5):
    l1 = tf.keras.losses.sparse_categorical_crossentropy(y1, y_hat1, from_logits=True)
    l2 = tf.keras.losses.sparse_categorical_crossentropy(y2, y_hat2, from_logits=True)

    return tf.expand_dims(l2, axis=1) + alpha * l1

def evaluate(sentiment_model, test_dataset):
    test_accuracy = tf.keras.metrics.BinaryAccuracy()

    for x, y1, y2 in test_dataset:
        # training=False is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        y_hat2, _ = sentiment_model(x, training=False)
        test_accuracy(y2, tf.math.sigmoid(y_hat2[:, -1, :]))

    acc = test_accuracy.result()
    print("Test set accuracy: {:.3%}".format(acc))

    return acc

def fit(clf_gpt2, train_dataset, test_dataset, params):
    optimizer = tf.keras.optimizers.Adam(learning_rate=params["lr"])

    # Keep results for plotting
    train_loss_results = []
    train_accuracy_results = []

    best_accuracy = 0
    for epoch in range(params["epochs"]):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()

        # Training loop - using batches of 32
        for step, (x, y1, y2) in enumerate(train_dataset):
            # Optimize the model
            with tf.GradientTape() as tape:
                # training=training is needed only if there are layers with different
                # behavior during training versus inference (e.g. Dropout).
                y_hat2, y_hat1 = clf_gpt2(x, training=True)
                loss_value = loss(y_hat1, y_hat2[:, -1, :], y1, y2, params["alpha"])

            grads = tape.gradient(loss_value, clf_gpt2.trainable_weights)
            optimizer.apply_gradients(zip(grads, clf_gpt2.trainable_weights))

            # Track loss progress
            epoch_loss_avg(loss_value)

            # Track accuracy progress
            y_hat2, _ = clf_gpt2(x, training=True)
            epoch_accuracy(y2, tf.math.softmax(y_hat2[:, -1, :]))

            # End epoch
            train_loss_results.append(epoch_loss_avg.result())
            train_accuracy_results.append(epoch_accuracy.result())

            if step % 50 == 0:
                print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch + 1,
                                                                            epoch_loss_avg.result(),
                                                                            epoch_accuracy.result()))

        # Evaluate
        current_accuracy = evaluate(clf_gpt2, test_dataset)
        if current_accuracy > best_accuracy:
            print('\nEpoch %03d: val_acc improved from %0.5f to %0.5f,'
                  ' saving model to %s' % (epoch + 1, best_accuracy, current_accuracy, params_ft["check"]))

            clf_gpt2.save_weights(params_ft["check"])
            best_accuracy = current_accuracy

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
    train_labelled_text = load_dataset(params["train"], vocab, params["seqlen"])
    test_labelled_text = load_dataset(params["test"], vocab, params["seqlen"])

    train_sent_dataset = build_dataset(train_labelled_text, params["batch"])
    test_sent_dataset = build_dataset(test_labelled_text, params["batch"])

    # Calculate vocab_size from char2idx dict
    vocab_size = len(vocab)

    # Create GPT2 languade model configuration
    clf_gpt2_config = tm.GPT2Config(vocab_size, params["seqlen"], params["n_ctx"], params["embed"],
                                                params["layers"], params["heads"],
                                                output_attentions=True, resid_pdrop=params["drop"],
                                                embd_pdrop=params["drop"], attn_pdrop=params["drop"])

    # Load pre-trained GPT2 without language model head
    clf_gpt2 = GPT2Classifier(clf_gpt2_config, num_labels=4)
    if params["finetune"]:
        ckpt = tf.train.Checkpoint(net=clf_gpt2)
        ckpt.restore(tf.train.latest_checkpoint(params["check"]))

    # Train pre-trained model as a classifier
    fit(clf_gpt2, train_sent_dataset, test_sent_dataset, params)
