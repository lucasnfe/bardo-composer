import argparse
import numpy as np

from data_dnd import *

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

def build_dataset_nbayes(episodes, vocabulary, context_size, test_ep):
    (X_train, Y_train), (X_test, Y_test) = build_dataset(episodes, vocabulary, context_size, test_ep)

    # Transform X_train using idf
    X_train = CountVectorizer(vocabulary=vocabulary).fit_transform(X_train)
    X_train = TfidfTransformer().fit_transform(X_train)

    # Transform X_test using idf
    X_test = CountVectorizer(vocabulary=vocabulary).fit_transform(X_test)
    X_test = TfidfTransformer().fit_transform(X_test)

    return (X_train, Y_train), (X_test, Y_test)

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='clf_nbayes.py')
    parser.add_argument('--data', type=str, required=True, help="Dnd data.")
    parser.add_argument('--ctx', type=int, default=10, help="Context window size.")
    opt = parser.parse_args()

    # Load episodes in a dictionary
    episodes = load_episodes(opt.data)

    # Build vocabulary from the episodes
    vocabulary = build_vocabulary(episodes)

    # Run leave-one-out cross validation
    accuracies = []

    for ep, _ in sorted(episodes.items()):
        # Build dataset
        (X_train, Y_train), (X_test, Y_test) = build_dataset_nbayes(episodes, vocabulary, opt.ctx, test_ep=ep)

        # Train naive bayes on the train set
        clf_nbayes = MultinomialNB().fit(X_train, Y_train)

        # Evaluate on the test set
        predicted = clf_nbayes.predict(X_test)
        acc = np.mean(predicted == Y_test)

        accuracies.append(acc)

    # Compute average accuracy
    avg_acc = sum(accuracies)/len(accuracies)

    print(accuracies)
    print(avg_acc)
