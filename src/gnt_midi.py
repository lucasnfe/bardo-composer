import os
import json
import argparse
import scipy as sp
import numpy as np
import tensorflow as tf

import clf_dnd.data_dnd as dnd
import clf_vgmidi.data_vgmidi as vg
import clf_vgmidi.midi.encoder as me

from gnt_utils import *
from clf_vgmidi.models import *
from generators.gnt_beam import *
from generators.gnt_baseline import *

EPISODE_CTX = 20
GENERATED_DIR = '../output'

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

def classify_story_emotion(S, X, ix_fst, ix_lst, tokenizer, clf_dnd_valence, clf_dnd_arousal):
    episode_sentences = []

    story_emotion = []
    for i in range(len(X)):
        episode_sentences.append(X[i])
        if i < ix_fst:
            continue

        if i >= ix_lst:
            break

        # Get sentence duration
        duration = S[i+1] - S[i]
        print(duration)

        # Slice sentences to form the current story context
        ctx_sentences = ' '.join(episode_sentences[-EPISODE_CTX:])

        # Get the emotion of the current story context
        ctx_emotion = classify_sentence_emotion(ctx_sentences, tokenizer, clf_dnd_valence, clf_dnd_arousal)
        print(ctx_sentences, discretize_emotion(ctx_emotion))

        story_emotion.append((duration,ctx_emotion))

    return story_emotion

def generate_music_with_emotion(story_emotion, generation_params, language_model, clf_vgmidi, idx2char):
    clf_vgmidi_valence, clf_vgmidi_arousal = clf_vgmidi

    # Copy init_tokens
    init_tokens = list(generation_params["init_tokens"])

    # Init episode score
    episode_tokens = list(generation_params["init_tokens"])

    last_emotion = np.array([0, 0])
    try:
        for sentence in story_emotion:
            duration, ctx_emotion = sentence

            print(last_emotion, discretize_emotion(ctx_emotion))

            # Restart init tokens when emotions are different
            if (discretize_emotion(ctx_emotion) != last_emotion).any():
                print("CHANGED EMOTION!")
                generation_params["init_tokens"] = list(init_tokens)
                generation_params["previous"] = None

            # Get sentence duration
            generation_params["length"] = duration
            generation_params["emotion"] = ctx_emotion
            print(duration, discretize_emotion(ctx_emotion))

            # Generate music for this current story context
            if generation_params["mode"] == "beam":
                ctx_tokens, ctx_prob, ctx_text = beam_search(generation_params, language_model, clf_vgmidi_valence, clf_vgmidi_arousal, idx2char)
            elif generation_params["mode"] == "baseline":
                ctx_tokens, ctx_prob, ctx_text, prev = baseline(generation_params, idx2char)
                generation_params["previous"] = prev

            # Get the emotion of the generated music
            music_emotion = classify_music_emotion(np.array([generation_params["init_tokens"] + ctx_tokens]), clf_vgmidi_valence, clf_vgmidi_arousal)
            print(generation_params["init_tokens"] + ctx_tokens, discretize_emotion(music_emotion), ctx_prob)

            # Remove init tokens before
            episode_tokens += ctx_tokens

            generation_params["init_tokens"] = ctx_tokens[-params["n_ctx"]:]
            last_emotion = discretize_emotion(ctx_emotion)

            print("==========", "\n")
    except KeyboardInterrupt:
        print("Exiting due to keyboard interrupt.")

    return episode_tokens

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='midi_generator.py')
    parser.add_argument('--mode', type=str, default="beam", help="Generation strategy.")
    parser.add_argument('--ep',   type=str, required=True, help="Dnd episode to score.")
    parser.add_argument('--fst',  type=int, default=0, help="Sentence index to start the score.")
    parser.add_argument('--lst',  type=int, required=False, help="Sentence index to end the score.")
    parser.add_argument('--init', type=str, default="", help="Seed to start music generation.")
    parser.add_argument('--topk', type=int, default=10, help="Top k tokens to consider when sampling.")
    parser.add_argument('--beam', type=int, default=3, help="Beam Size.")

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

    # Load episodes (S is list of starting times and X is a list of sentences)
    S,X,_,_ = dnd.load_episode(opt.ep)

    # Load vgmidi pieces for baseline
    vgmidi = vg.load_dataset("../data/vgmidi/vgmidi_bardo_train.csv", vocab, params["seqlen"], dimesion=2, splits=["split_1"])

    # Set first and last indices of sentences in the story
    ix_fst = opt.fst
    if opt.lst == None:
        ix_lst = len(X) - 1
    else:
        ix_lst = opt.lst

    # Compute emotion in the given story using dnd classifier
    tokenizer = tm.BertTokenizer.from_pretrained('bert-base-uncased')

    # Load sentence classifier
    clf_dnd_valence = load_clf_dnd(vocab_size, "../trained/clf_bert.ckpt/clf_bert_ep3_0/clf_bert")
    clf_dnd_arousal = load_clf_dnd(vocab_size, "../trained/clf_bert.ckpt/clf_bert_ep3_1/clf_bert")

    story_emotion = classify_story_emotion(S, X, ix_fst, ix_lst, tokenizer, clf_dnd_valence, clf_dnd_arousal)

    # Clean up story classifier
    del tokenizer
    del clf_dnd_valence
    del clf_dnd_arousal

    tf.keras.backend.clear_session()

    # Create idx2char from char2idx dict
    idx2char = {idx:char for char,idx in vocab.items()}

    # Encode init text as sequence of indices
    init_music = preprocess_text(opt.init)
    init_tokens = [vocab[word] for word in init_music.split(" ")]

    # Generation parameters
    generation_params = {"init_tokens"  : init_tokens,
                          "vocab_size"  : vocab_size,
                               "vgmidi" : vgmidi,
                               "mode"   : opt.mode,
                               "top_k"  : opt.topk,
                          "beam_width"  : opt.beam,
                            "previous"  : None,
                               "n_ctx"  : params["n_ctx"]}

    # Load generative language model
    language_model = load_language_model(vocab_size, params, "../trained/transformer.ckpt")

    # Load generative language model
    clf_vgmidi_valence = load_clf_vgmidi(vocab_size, params, "../trained/clf_gpt2.ckpt/clf_gpt2_0/clf_gpt2")
    clf_vgmidi_arousal = load_clf_vgmidi(vocab_size, params, "../trained/clf_gpt2.ckpt/clf_gpt2_1/clf_gpt2")

    episode_tokens = generate_music_with_emotion(story_emotion, generation_params, language_model,
                                                 (clf_vgmidi_valence, clf_vgmidi_arousal), idx2char)

    # Decode generated tokens
    episode_score = " ".join([idx2char[idx] for idx in episode_tokens])
    print(episode_score)

    # Write piece as midi and wav
    episode_name = os.path.splitext(os.path.basename(opt.ep))[0]
    piece_path = os.path.join(GENERATED_DIR, episode_name + "_score")
    me.write(episode_score, piece_path)
