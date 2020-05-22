import argparse
import tensorflow as tf

from data_dnd import *

BUFFER_SIZE=10000

def build_dataset_lstm(episodes, vocabulary, context_size, batch_size, test_ep):
    (X_train, Y_train), (X_test, Y_test) = build_dataset(episodes, vocabulary, context_size, test_ep)

    train_dataset = []
    for i in range(len(X_train)):
        # Tokenize train sentence
        X_train[i] = [vocabulary[w] for w in X_train[i].split()]

        # add tokenized sentence to the train dataset
        train_dataset.append((X_train[i], [Y_train[i]]))

    train_dataset = tf.data.Dataset.from_generator(lambda: train_dataset, (tf.int32, tf.int32))
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.padded_batch(batch_size, padded_shapes=([None], [None]))

    test_dataset = []
    for i in range(len(X_test)):
        # Tokenize test sentence
        X_test[i] = [vocabulary[w] for w in X_test[i].split()]

        # add tokenized sentence to the test dataset
        test_dataset.append((X_test[i], [Y_test[i]]))

    test_dataset = tf.data.Dataset.from_generator(lambda: test_dataset, (tf.int32, tf.int32))
    test_dataset = test_dataset.padded_batch(batch_size, padded_shapes=([None], [None]))

    return train_dataset, test_dataset

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='midi_generator.py')
    parser.add_argument('--data', type=str, required=True, help="Dnd data.")
    parser.add_argument('--ctx', type=int, default=10, help="Context window size.")
    parser.add_argument('--batch', type=int, default=32, help="Context window size.")
    opt = parser.parse_args()

    # Load episodes in a dictionary
    episodes = load_episodes(opt.data)

    # Build vocabulary from the episodes
    vocabulary = build_vocabulary(episodes)

    # Run leave-one-out cross validation
    accuracies = []

    for ep, _ in sorted(episodes.items()):
        # Build dataset
        train_dataset, test_dataset = build_dataset_lstm(episodes, vocabulary, opt.ctx, opt.batch, test_ep=ep)

        clf_lstm = tf.keras.Sequential([
            tf.keras.layers.Embedding(len(vocabulary), 64),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        clf_lstm.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                    optimizer=tf.keras.optimizers.Adam(1e-4), metrics=['accuracy'])

        history = clf_lstm.fit(train_dataset, epochs=10, validation_data=test_dataset)