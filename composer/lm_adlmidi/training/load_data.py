import os
import hashlib

import tensorflow as tf

def midi2text_paths(midi_paths):
    dataset_txt_files = []

    for midi_file in midi_paths:
        txt_file = os.path.splitext(midi_file)[0] + ".txt"
        if os.path.exists(txt_file):
            dataset_txt_files.append(txt_file)

    return dataset_txt_files

def dataset_difference(origin, dest):
    origin_files = []

    dest_md5 = {}
    for file in dest:
        with open(file, "rb") as midi_file:
            midi_md5 = hashlib.md5(midi_file.read()).hexdigest()
            dest_md5[midi_md5] = file

    for file in origin:
        with open(file, "rb") as midi_file:
            midi_md5 = hashlib.md5(midi_file.read()).hexdigest()
            if midi_md5 not in dest_md5:
                origin_files.append(file)

    return origin_files

def load_dataset(dir_path, extensions=[]):
    files = []
    for file in os.listdir(dir_path):
        filename, ext = os.path.splitext(file)
        if ext in extensions:
            file_path = os.path.join(dir_path, file)
            files.append(file_path)

    return files

def build_dataset(dataset_midi_files, vocab, params):
    global TF_VOCAB, SEQ_LENGTH, BATCH_SIZE
    
    # Set global parameters
    TF_VOCAB = build_tf_vocab(vocab)
    SEQ_LENGTH = params["seqlen"]
    BATCH_SIZE = params["batch"]

    # Get a list of the txt files associated to the midi files
    dataset_txt_files = midi2text_paths(dataset_midi_files)

    # Read all files in the dataset directory and combine them
    list_ds = tf.data.Dataset.from_tensor_slices(dataset_txt_files)
    list_ds = list_ds.prefetch(tf.data.experimental.AUTOTUNE)

    dataset = list_ds.flat_map(_process_path)

    return dataset

def _process_path(filepath):
    # Split content by space to get words
    text = tf.io.read_file(filepath)
    words = tf.strings.split(text, sep=" ")

    # Tokenize every word in text
    tokens = tf.map_fn(lambda t: TF_VOCAB.lookup(t), words, dtype=tf.int32)
    tokens = tf.data.Dataset.from_tensor_slices(tokens)

    # Batchify tokens
    sequences = tokens.batch(SEQ_LENGTH + 1, drop_remainder=True)
    dataset = sequences.map(_split_input_target)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    return dataset

def _split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

def build_tf_vocab(vocab):
    keys = list(vocab.keys())
    values = list(vocab.values())

    # build a tf lookup table
    tf_vocab_init = tf.lookup.KeyValueTensorInitializer(keys=tf.constant(keys), values=tf.constant(values))
    TF_VOCAB = tf.lookup.StaticHashTable(initializer=tf_vocab_init, default_value=tf.constant(-1), name="vocab")

    return TF_VOCAB
