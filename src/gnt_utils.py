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

def concat_all_tokens(tokens, range, gen_len):
    all_tokens = tf.range(range)
    all_tokens = tf.reshape(all_tokens, (range, 1))

    tokens = tf.tile(tokens, tf.constant([range, 1], tf.int32))
    tokens = tf.concat((tokens, all_tokens), axis=1)

    tokens = tf.pad(tokens, tf.constant([[0, 0,], [0, gen_len - tokens.shape[-1]]]), "CONSTANT", constant_values=1)

    return tokens

def run_language_model(init_tokens, language_model):
    generative_x = init_tokens[:, -n_ctx:]
    generative_y = language_model(generative_x, training=False)
    generative_p = tf.math.softmax(generative_y).numpy().squeeze()

    return generative_p

def classify_story_emotion(story_x, tokenizer, clf_dnd_valence, clf_dnd_arousal):
    story_tokens = tf.constant(tokenizer.encode(story_x, add_special_tokens=True))[None, :]

    story_valence = clf_dnd_valence(story_tokens)
    story_arousal = clf_dnd_arousal(story_tokens)
    story_emotion = tf.math.sigmoid(tf.concat([story_valence, story_arousal], 1)).numpy().squeeze()

    return story_emotion

def classify_music_emotion(music_x, story_emotion, clf_vgmidi_valence, clf_vgmidi_arousal):
    music_valence = clf_vgmidi_valence(music_x, training=False)
    music_arousal = clf_vgmidi_arousal(music_x, training=False)

    music_emotion = tf.math.sigmoid(tf.concat([music_valence, music_arousal], 1)).numpy().squeeze()

    music_valence = music_emotion[:,0]
    if story_emotion[0] < 0.5:
        music_valence = 1.0 - music_valence

    music_arousal = music_emotion[:,1]
    if story_emotion[1] < 0.5:
        music_arousal = 1.0 - music_arousal

    return music_valence, music_arousal
