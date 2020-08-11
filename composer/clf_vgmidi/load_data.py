import os
import csv
import tensorflow as tf

BUFFER_SIZE=10000

def build_dataset(datapath, vocab, seq_length, batch_size, dimesion=0, splits=["split_1", "split_2", "split_4", "split_8", "split_16"]):
    dataset = []

    data = csv.DictReader(open(datapath, "r"))

    midi_paths = []
    for row in data:
        filepath, valence, arousal = row["filepath"], int(row["valence"]), int(row["arousal"])
        for split in splits:
            if split + "/" in filepath:
                # Form midi filepath
                piece_path = os.path.join(os.path.dirname(datapath), filepath)

                # Form txt filepath
                txt_file = os.path.splitext(piece_path)[0] + ".txt"

                if os.path.exists(txt_file):

                    # Read txt file
                    tokens = []
                    with open(txt_file) as fp:
                        tokens = [vocab[w] for w in fp.read().split(" ")]

                    if dimesion == 0:
                        label = [valence]
                    elif dimesion == 1:
                        label = [arousal]
                    elif dimesion == 2:
                        label = [valence, arousal]

                    dataset.append((tokens, label))

    tf_dataset = tf.data.Dataset.from_generator(lambda: dataset, (tf.int32, tf.int32))
    tf_dataset = tf_dataset.shuffle(BUFFER_SIZE)
    tf_dataset = tf_dataset.padded_batch(batch_size, padded_shapes=([None], [1]), padding_values=(1, 1))

    return tf_dataset
