import os
import csv
import json
import argparse

import numpy as np
import tensorflow as tf
import transformers as tm
import midi.encoder as me

BUFFER_SIZE=10000

class GPT2Classifier(tm.modeling_tf_gpt2.TFGPT2Model):
    def __init__(self, config, num_labels=1, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # self.dropout = tf.keras.layers.Dropout(0.5)
        self.emotion_head = tf.keras.layers.Dense(num_labels)

    def call(self, inputs, **kwargs):
        # Extract features
        outputs = super().call(inputs, **kwargs)

        #  Extract language model logits
        # lm_logits = self.transformer.wte(outputs[0], mode="linear", training=kwargs["training"])

        # Finetuner Emotion Head
        emotion_logits = self.emotion_head(outputs[0], training=kwargs["training"])

        return emotion_logits

def load_dataset(datapath, vocab, seq_length):
    dataset = []

    data = csv.DictReader(open(datapath, "r"))
    for row in data:
        filepath, label = row["filepath"], row["label"]

        piece_path = os.path.join(os.path.dirname(datapath), filepath)
        piece_text = me.load_file(piece_path).split(" ")
        tokens = [vocab[c] for c in piece_text]

        dataset.append((tokens, [label]))

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
    clf_gpt2_config = tm.GPT2Config(vocab_size, params["seqlen"], params["n_ctx"], params["embed"],
                                                params["layers"], params["heads"],
                                                output_attentions=True, resid_pdrop=params["drop"],
                                                embd_pdrop=params["drop"], attn_pdrop=params["drop"])

    # Load pre-trained GPT2 without language model head
    clf_gpt2 = GPT2Classifier(clf_gpt2_config, num_labels=4)
    if params["finetune"]:
        ckpt = tf.train.Checkpoint(net=clf_gpt2)
        ckpt.restore(tf.train.latest_checkpoint(params["pretr"]))

    clf_gpt2.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(params["lr"]), metrics=['accuracy'])

    history = clf_gpt2.fit(train_dataset, epochs=10, validation_data=test_dataset)
