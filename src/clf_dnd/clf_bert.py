import argparse
import tensorflow as tf
import transformers as tm

from data_dnd import *

BUFFER_SIZE=10000

from models import *

def build_dataset_bert(episodes, vocabulary, context_size, batch_size, test_ep, pre_trained=False, dimesion=0):
    (X_train, Y1_train, Y2_train), (X_test, Y1_test, Y2_test) = build_dataset(episodes, vocabulary, context_size, test_ep)

    if pre_trained:
        tokenizer = tm.BertTokenizer.from_pretrained('bert-base-uncased')

    train_examples = []
    for i in range(len(X_train)):
        # Tokenize train sentence
        if pre_trained:
            X_train[i] = tokenizer.encode(X_train[i], add_special_tokens=True)
        else:
            X_train[i] = [vocabulary[w] for w in X_train[i].split()]

        # add tokenized sentence to the train dataset
        if dimesion == 0:
            Y_train = Y1_train[i]
        elif dimesion == 1:
            Y_train = Y2_train[i]

        train_examples.append((X_train[i], [Y_train]))

    train_dataset = tf.data.Dataset.from_generator(lambda: iter(train_examples), (tf.int32, tf.int32))
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.padded_batch(batch_size, padded_shapes=([None], [1]))

    test_examples = []
    for i in range(len(X_test)):
        # Tokenize test sentence
        if pre_trained:
            X_test[i] = tokenizer.encode(X_test[i], add_special_tokens=True)
        else:
            X_test[i] = [vocabulary[w] for w in X_test[i].split()]

        # add tokenized sentence to the test dataset
        if dimesion == 0:
            Y_test = Y1_test[i]
        elif dimesion == 1:
            Y_test = Y2_test[i]

        test_examples.append((X_test[i], [Y_test]))

    test_dataset = tf.data.Dataset.from_generator(lambda: iter(test_examples), (tf.int32, tf.int32))
    test_dataset = test_dataset.padded_batch(batch_size, padded_shapes=([None], [1]))

    return train_dataset, test_dataset

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='clf_bert.py')
    parser.add_argument('--data', type=str, required=True, help="Dnd data.")
    parser.add_argument('--ctx', type=int, default=10, help="Context window size.")
    parser.add_argument('--batch', type=int, default=32, help="Batch size.")
    parser.add_argument('--dim', type=int, default=0, help="Dimension to train.")
    parser.add_argument('--pre', dest='pre', action='store_true')
    parser.set_defaults(pre=False)
    opt = parser.parse_args()

    # Load episodes in a dictionary
    episodes = load_episodes(opt.data)

    # Build vocabulary from the episodes
    vocabulary = build_vocabulary(episodes)

    # Run leave-one-out cross validation
    accuracies = []

    for ep, _ in sorted(episodes.items()):
        print("------> Leave-one-out", ep)

        # Build dataset
        train_dataset, test_dataset = build_dataset_bert(episodes, vocabulary, opt.ctx, opt.batch, test_ep=ep, pre_trained=opt.pre, dimesion=opt.dim)

        if opt.pre:
            clf_transf = Bert.from_pretrained('bert-base-uncased', num_labels=1)
        else:
            clf_config = tm.BertConfig(len(vocabulary), hidden_size=256, num_hidden_layers=2, num_attention_heads=8, num_labels=1)
            clf_transf = Bert(clf_config)

        clf_transf.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    optimizer=tf.keras.optimizers.Adam(1e-4), metrics=['accuracy'])

        checkpoint = tf.keras.callbacks.ModelCheckpoint('../../trained/clf_bert.ckpt/clf_bert_' + ep + "_" + str(dim) + "/clf_bert",
            monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True)

        history = clf_transf.fit(train_dataset, epochs=10, validation_data=test_dataset, callbacks=[checkpoint])

        # Clear the keras session
        tf.keras.backend.clear_session()

        # Discard previous clf_transf
        del clf_transf
