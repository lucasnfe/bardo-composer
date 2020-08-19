import os
import string
import unidecode
from word2number import w2n

STOPWORDS = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
             "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
             "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
             "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
             "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has",
             "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but",
             "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with",
             "about", "against", "between", "into", "through", "during", "before", "after",
             "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
             "under", "again", "further", "then", "once", "here", "there", "when", "where",
             "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
             "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
             "very", "s", "t", "can", "will", "just", "don", "should", "now"]

LABEL_MAP = {"happy": 0, "agitated": 1, "suspense": 2, "calm": 3}

def remove_stopwords(text):
    """Remove stop words"""
    # Remove stop words
    for stopword in STOPWORDS:
        text = text.replace(' ' + stopword + ' ', ' ')

    return text

def remove_accented_chars(text):
    """Remove accented characters from text, e.g. café"""
    text = unidecode.unidecode(text)
    return text

def remove_punctuation(text):
    """Remove punctuation"""
    table = str.maketrans('', '', string.punctuation)
    text = [w.translate(table) for w in text.split(" ")]
    return " ".join(text)

def convert_numbers(text):
    """Convert numbers to number words"""
    tokens = []
    for token in text.split(" "):
        try:
            word = w2n.num_to_word(token)
            tokens.append(word)
        except:
            tokens.append(token)

    return " ".join(tokens)

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

                    # Convert to lower case
                    x = x.lower()

                    # Text pre-processing
                    x = remove_accented_chars(x)
                    x = remove_stopwords(x)
                    x = remove_punctuation(x)
                    x = convert_numbers(x)

                    episodes[fname]["X"].append(x)
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
            sentence_with_context = sentences[i - context_size:i+1]

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