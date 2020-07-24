import json
import argparse
import tensorflow as tf

from training.load_data import *
from training.train import *

def build_language_model(vocab_size, params):
    # Build Transformer for Language Modeling task
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Embedding(vocab_size, params["embed"], batch_input_shape=[params["batch"], None]))

    for i in range(max(1, params["layers"])):
        model.add(tf.keras.layers.LSTM(params["hidden"], return_sequences=True, stateful=True, dropout=params["drop"], recurrent_dropout=params["drop"]))

    model.add(tf.keras.layers.Dense(vocab_size))

    return model

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='gen_lstm.py')
    parser.add_argument('--conf', type=str, required=True, help="JSON file with training parameters.")
    opt = parser.parse_args()

    # Load training parameters
    params = {}
    with open(opt.conf) as conf_file:
        params = json.load(conf_file)["lm_lstm"]

    # Load vocab from json file
    vocab = {}
    with open(params["vocab"]) as f:
        vocab = json.load(f)

    # Calculate vocab_size from char2idx dict
    vocab_size = len(vocab)

    train_midi_files = load_dataset(params["train"], extensions=[".mid", ".midi"])
    test_midi_files = load_dataset(params["test"], extensions=[".mid", ".midi"])

    # Load training files, skipping the ones in skip_files.
    if "skip" in params:
        skip_midi_files = load_dataset(params["skip"], extensions=[".mid", ".midi"])
        train_midi_files = dataset_difference(train_midi_files, skip_midi_files)
        test_midi_files = dataset_difference(test_midi_files, skip_midi_files)

    # Build dataset from encoded unlabelled midis
    train_dataset = build_dataset(train_midi_files, vocab, params)
    test_dataset = build_dataset(test_midi_files, vocab, params)

    # Train Transformer GPT
    n_train_steps = calc_steps(train_midi_files, params["seqlen"], params["batch"])
    print("n_train_steps:", n_train_steps)

    # Build lstm language model
    gen_lstm = build_language_model(vocab_size, params)

    # Train lstm language model
    train_language_model(gen_lstm, params, train_dataset, test_dataset, n_train_steps)
