import os
import csv
import tensorflow as tf
import clf_vgmidi.midi.encoder as me

BUFFER_SIZE=10000

def load_dataset(datapath, vocab, seq_length, dimesion=0):
    dataset = []

    data = csv.DictReader(open(datapath, "r"))
    for row in data:
        filepath, valence, arousal = row["filepath"], int(row["valence"]), int(row["arousal"])

        piece_path = os.path.join(os.path.dirname(datapath), filepath)
        piece_text = me.load_file(piece_path).split(" ")
        tokens = [vocab[c] for c in piece_text]

        if dimesion == 0:
            label = valence
        elif dimesion == 1:
            label = arousal

        dataset.append((tokens[:seq_length], [label]))

    return dataset

def build_dataset(dataset, batch_size):
    # Read all files in the dataset directory and combine them
    tf_dataset = tf.data.Dataset.from_generator(lambda: dataset, (tf.int32, tf.int32))
    tf_dataset = tf_dataset.shuffle(BUFFER_SIZE)
    tf_dataset = tf_dataset.padded_batch(batch_size, padded_shapes=([None], [1]), padding_values=(1, 1))

    return tf_dataset
