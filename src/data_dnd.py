import os

LABEL_MAP = {"calm": 0, "happy": 1, "agitated": 2, "suspense": 3}

def load_episodes(dirpath):
    episodes = {}

    for dir, _ , files in os.walk(dirpath):
        for i, fname in enumerate(files):
            # Each file is a episode
            episodes[fname] = {"X": [], "Y": []}

            filepath = os.path.join(dir, fname)
            with open(filepath) as f:
                for line in f:
                    example = line.rsplit(":", 1)
                    x, y = example[0], example[1].replace('\n', '')
                    episodes[fname]["X"].append(x.lower())
                    episodes[fname]["Y"].append(LABEL_MAP[y])

    return episodes

def build_vocabulary(episodes):
    sentences = ""
    for ep in episodes:
        for x in episodes[ep]["X"]:
            sentences += (x + " ").lower()

    words = set(sentences.split())
    vocab = {w:i for i,w in enumerate(words)}

    return vocab

def parse_contexts(sentences, context_size):
    contexts = []
    for i in range(len(sentences)):
        if i < context_size:
            sentence_with_context = sentences[:i+1]
        else:
            sentence_with_context = sentences[i - context_size:i]

        contexts.append(' '.join(sentence_with_context))

    return contexts

def build_dataset(episodes, vocabulary, context_size=30, test_ep="ep1.txt"):
    train_episodes = dict(episodes)
    test_episode = train_episodes.pop(test_ep)

    X_train, Y_train = [],[]

    # Build train set
    for ep in train_episodes:
        X_train += parse_contexts(train_episodes[ep]["X"], context_size)
        Y_train += train_episodes[ep]["Y"]

    # Build test set
    X_test = parse_contexts(test_episode["X"], context_size)
    Y_test = test_episode["Y"]

    return (X_train, Y_train), (X_test, Y_test)