import os
import json
import argparse
import scipy as sp
import numpy as np
import tensorflow as tf
import clf_vgmidi.midi.encoder as me

from gnt_utils import *
from clf_vgmidi.models import *
from gnt_beam.beam_search import *

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
                          "vocab_size": vocab_size,
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
