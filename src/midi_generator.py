import os
import json
import argparse
import scipy as sp
import numpy as np
import tensorflow as tf
import clf_vgmidi.midi.encoder as me

from clf_vgmidi.models import *
from gnt_beam.beam_search import *

GENERATED_DIR = '../output'

def preprocess_text(text):
    if text == "":
        tokens = "\n"
    else:
        tokens = text.split(" ")

    # Remove \n
    tokens = list(filter(lambda t: t != "\n", tokens))
    tokens = ["\n"] + tokens

    return " ".join(tokens)

def load_language_model(vocab_size, params, path):
    # Instanciate GPT2 language model
    gpt2_config = tm.GPT2Config(vocab_size, params["seqlen"], params["n_ctx"], params["embed"], params["layers"], params["heads"],
                               resid_pdrop=params["drop"], embd_pdrop=params["drop"], attn_pdrop=params["drop"])

    # Load GPT2 language model trained weights
    language_model = GPT2LanguadeModel(gpt2_config)
    ckpt = tf.train.Checkpoint(net=language_model)
    ckpt.restore(tf.train.latest_checkpoint(path)).expect_partial()

    return language_model

def load_clf_dnd(vocab_size, path):
    # Load Bert trained weights
    clf_dnd = tm.TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
    clf_dnd.load_weights(path).expect_partial()

    return clf_dnd

def load_clf_vgmidi(vocab_size, params, path="../trained/clf_vgmidi.ckpt"):
    # Instanciate GPT2 music emotion classifier
    gpt2_config = tm.GPT2Config(vocab_size, params["seqlen"], params["n_ctx"], params["embed"], params["layers"], params["heads"],
                               resid_pdrop=params["drop"], embd_pdrop=params["drop"], attn_pdrop=params["drop"])

    # Load pre-trained GPT2 without language model head
    clf_vgmidi = GPT2Classifier(gpt2_config)
    clf_vgmidi.load_weights(path).expect_partial()

    return clf_vgmidi

def classify_story_emotion(story_x, tokenizer, clf_dnd_valence, clf_dnd_arousal):
    story_tokens = tf.constant(tokenizer.encode(story_x, add_special_tokens=True))[None, :]

    story_valence = clf_dnd_valence(story_tokens)
    story_arousal = clf_dnd_arousal(story_tokens)
    story_emotion = tf.math.sigmoid(tf.concat([story_valence, story_arousal], 1)).numpy().squeeze()

    return story_emotion

def classify_music_emotion(music_x, clf_vgmidi_valence, clf_vgmidi_arousal):
    music_valence = clf_vgmidi_valence(music_x, training=False)
    music_arousal = clf_vgmidi_arousal(music_x, training=False)

    music_emotion = tf.math.sigmoid(tf.concat([music_valence, music_arousal], 1)).numpy().squeeze()

    return music_emotion

def generate_midi(generation_params, language_model, clf_vgmidi_valence, clf_vgmidi_arousal, tokenizer):
    story_emotion = generation_params["emotion"]
    init_tokens   = generation_params["init_tokens"]
    gen_len       = generation_params["length"]
    n_ctx         = generation_params["n_ctx"]
    top_k         = generation_params["top_k"]

    generated = np.array([init_tokens])

    while generated.shape[-1] < gen_len:
        # Run language_model
        generative_x = generated[:, -n_ctx:]
        generative_y = language_model(generative_x, training=False)
        generative_p = tf.math.softmax(generative_y)

        # Get topk most likely tokens from the language model
        top_probs, top_tokens = tf.math.top_k(generative_p, top_k)

        # Create tensor with all possible tokens
        top_tokens_tiled = np.reshape(top_tokens, (top_tokens.shape[-1], 1))
        generated_tiled = np.tile(generated, top_tokens_tiled.shape)

        # Concatenate sequence of tokens generated so far with all possible tokens
        music_x = np.concatenate((generated_tiled, top_tokens_tiled), axis=1)
        music_x = np.pad(music_x, ((0, 0), (0, gen_len - music_x.shape[-1])), 'constant', constant_values=(1, 1))

        # Classifiy music emotion considering all possible tokens
        music_emotion = classify_music_emotion(music_x, clf_vgmidi_valence, clf_vgmidi_arousal)

        print("Music Emotion:", music_emotion)
        music_valence = music_emotion[:,0]
        if story_emotion[0] < 0.5:
            music_valence = 1.0 - music_valence

        music_arousal = music_emotion[:,1]
        if story_emotion[1] < 0.5:
            music_arousal = 1.0 - music_arousal

        top_probs = top_probs.numpy().squeeze()
        top_tokens = top_tokens.numpy().squeeze()

        final_logp = tf.math.log(top_probs * music_valence * music_arousal)

        ix = tf.random.categorical([final_logp], 1)
        predicted_token = top_tokens[int(ix)]

        # Append predicted_id to generated midi
        predicted_token = np.reshape(predicted_token, (1, 1))
        generated = np.concatenate((generated, predicted_token), axis=1)

    midi_text = list(generated.squeeze())

    return list(generated.squeeze())

if __name__ == "__main__":
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser(description='midi_generator.py')
    parser.add_argument('--mode', type=str, default="sample", help="Generation strategy.")
    parser.add_argument('--init', type=str, required=True, help="Seed text to start generation.")
    parser.add_argument('--text', type=str, required=True, help="Seed text to start generation.")
    parser.add_argument('--glen', type=int, default=256, help="Length of generated midi.")
    parser.add_argument('--topk', type=int, default=10, help="Top k tokens to consider when sampling.")

    opt = parser.parse_args()

    # Load training parameters
    params = {}
    with open("clf_vgmidi/clf_gpt2_conf.json") as conf_file:
        params = json.load(conf_file)["clf_gpt2"]

    # Load char2idx dict from json file
    with open("../trained/vocab.json") as f:
        vocab = json.load(f)

    # Calculate vocab_size from char2idx dict
    vocab_size = len(vocab)

    # Load generative language model
    language_model = load_language_model(vocab_size, params, "../trained/transformer.ckpt")

    # Load generative language model
    clf_vgmidi_valence = load_clf_vgmidi(vocab_size, params, "../trained/clf_gpt2.ckpt/clf_gpt2_0/clf_gpt2")
    clf_vgmidi_arousal = load_clf_vgmidi(vocab_size, params, "../trained/clf_gpt2.ckpt/clf_gpt2_1/clf_gpt2")

    # Load generative language model
    clf_dnd_valence = load_clf_dnd(vocab_size, "../trained/clf_bert.ckpt/clf_bert_ep1_0/clf_bert")
    clf_dnd_arousal = load_clf_dnd(vocab_size, "../trained/clf_bert.ckpt/clf_bert_ep1_1/clf_bert")

    # Encode init text as sequence of indices
    init_music = preprocess_text(opt.init)
    init_tokens = [vocab[word] for word in init_music.split(" ")]

    # Compute emotion in the given story using dnd classifier
    tokenizer = tm.BertTokenizer.from_pretrained('bert-base-uncased')
    story_emotion = classify_story_emotion(opt.text, tokenizer, clf_dnd_valence, clf_dnd_arousal)
    print("story_emotion", story_emotion)

    # Generation parameters
    generation_params = {"init_tokens": init_tokens,
                              "length": opt.glen,
                               "top_k": opt.topk,
                               "n_ctx": params["n_ctx"],
                             "emotion": story_emotion}

    # Generate a midi as text
    generated_tokens = beam_search(generation_params, language_model, clf_vgmidi_valence, clf_vgmidi_arousal, tokenizer)

    # Create idx2char from char2idx dict
    idx2char = {idx:char for char,idx in vocab.items()}

    # Decode generated tokens
    generated_text = " ".join([idx2char[idx] for idx in generated_tokens])

    # Save independent pieces.
    for i, piece_text in enumerate(generated_text.split("\n")):
        piece_text = piece_text.strip()
        if len(piece_text) > 0:
            print(piece_text)

            # Write piece as midi and wav
            piece_path = os.path.join(GENERATED_DIR, "generated_" + str(i))
            me.write(piece_text, piece_path)
