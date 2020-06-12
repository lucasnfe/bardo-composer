import numpy as np
import tensorflow as tf

def preprocess_text(text):
    if text == "":
        tokens = "\n"
    else:
        tokens = text.split(" ")

    # Remove \n
    tokens = list(filter(lambda t: t != "\n", tokens))
    tokens = ["\n"] + tokens

    return " ".join(tokens)

def concat_all_tokens(tokens, shape, gen_len):
    all_tokens = tf.range(shape[0])
    all_tokens = tf.reshape(all_tokens, (shape[0], 1))

    all_tokens = tf.tile(all_tokens, tf.constant([shape[1], 1], tf.int32))

    tokens = tf.concat((tokens, all_tokens), axis=1)
    #tokens = tf.pad(tokens, tf.constant([[0, 0,], [0, gen_len - tokens.shape[-1]]]), "CONSTANT", constant_values=1)

    return tokens

def run_language_model(init_tokens, language_model, n_ctx):
    with tf.device('/GPU:0'):
        generative_x = init_tokens[:, -n_ctx:]
        generative_y = language_model(generative_x, training=False)
        #generative_y = tf.math.softmax(generative_y).numpy().squeeze()

    return generative_y

def classify_sentence_emotion(story_x, tokenizer, clf_dnd_valence, clf_dnd_arousal):
    # Create batch of size 1
    story_tokens = tf.expand_dims(tokenizer.encode(story_x, add_special_tokens=True), 0)

    with tf.device('/GPU:0'):
        story_valence = clf_dnd_valence(story_tokens, training=False)
    with tf.device('/GPU:1'):
        story_arousal = clf_dnd_arousal(story_tokens, training=False)

    story_emotion = tf.math.sigmoid(tf.concat([story_valence, story_arousal], 1)).numpy().squeeze()

    return story_emotion

def classify_music_emotion(music, clf_vgmidi_valence, clf_vgmidi_arousal):
    music_x = np.array(music)
    if music_x.shape[-1] < 32:
        music_x = np.pad(music_x, ((0, 0), (0, 32)), 'constant', constant_values=(1, 1))

    with tf.device('/GPU:0'):
        music_valence = clf_vgmidi_valence(music_x, training=False)

    with tf.device('/GPU:1'):
        music_arousal = clf_vgmidi_arousal(music_x, training=False)

    music_emotion = tf.math.sigmoid(tf.concat([music_valence, music_arousal], 1)).numpy().squeeze()

    return music_emotion

def compute_music_emotion_probability(music_x, story_emotion, clf_vgmidi_valence, clf_vgmidi_arousal):
    music_emotion = classify_music_emotion(music_x, clf_vgmidi_valence, clf_vgmidi_arousal)

    music_valence = music_emotion[:,0]
    if story_emotion[0] < 0.5:
        music_valence = 1.0 - music_valence

    music_arousal = music_emotion[:,1]
    if story_emotion[1] < 0.5:
        music_arousal = 1.0 - music_arousal

    return music_valence, music_arousal


def discretize_emotion(emotion):
    discrete_emotion = np.array([0, 0])
    if emotion[0] > 0.5:
        discrete_emotion[0] = 1
    if emotion[1] > 0.5:
        discrete_emotion[1] = 1

    return discrete_emotion

def sample_without_replacement(logits, n_samples):
    drawn_samples = []

    distribution = np.array(logits)
    while len(drawn_samples) < n_samples:
        s = int(tf.random.categorical([distribution], 1))
        drawn_samples.append(s)

        # Remove s from distribution
        distribution = np.delete(distribution, s, 0)
    return drawn_samples

def compute_penalty(tokens_so_far, generative_p, pen_len, alpha=0.25, dont_penalize=[0,1]):
    # Count occurences of each element
    tokens_to_penalize = tokens_so_far[:, -pen_len:]
    count = np.apply_along_axis(lambda x: np.bincount(x, minlength=generative_p.shape[-1]), axis=1, arr=tokens_to_penalize)

    # Compute penalty per token
    penalty = np.power(np.array(alpha), count)

    # Penalty has to be it's reciprocal for logits that are greater than zero
    # penalty = np.reciprocal(penalty, out=penalty, where=(generative_y > 0))

    # Don't penalize tokens that are in dont_penalize list
    indices_i = np.arange(tokens_so_far.shape[0]).reshape(tokens_so_far.shape[0], 1)
    indices_j = dont_penalize
    penalty[indices_i, indices_j] = 1.0
    # penalty = penalty.squeeze()

    return penalty

def load_vgmidi_pieces_with_emotion(vgmidi, emotion):
    pieces_with_emotion = []
    for p,e in vgmidi:
        if (np.array(e) == discretize_emotion(emotion)).all():
            pieces_with_emotion.append((p, e))

    print("Found", len(pieces_with_emotion), "with emotion", discretize_emotion(emotion))
    return pieces_with_emotion

def get_rand_prefix_with_emotion(vgmidi, emotion, prefix_len=16):
    # Load all pieces in the vgmidi dataset with the desired emotion
    pieces_with_emotion = load_vgmidi_pieces_with_emotion(vgmidi, emotion)
    rand_ix = np.random.randint(len(pieces_with_emotion))

    rand_piece, _ = pieces_with_emotion[rand_ix] 

    return rand_piece[:prefix_len]
