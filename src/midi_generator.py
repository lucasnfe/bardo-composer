import json
import argparse
import numpy as np
import tensorflow as tf

from clf_dnd.models import *
from clf_vgmidi.models import *

def preprocess_text(text):
    if text == "":
        tokens = "\n"
    else:
        tokens = text.split(" ")

    # Remove \n
    tokens = list(filter(lambda t: t != "\n", tokens))

    # Add \n
    tokens = ["\n"] + tokens

    return " ".join(tokens)

def load_language_model(vocab_size, params, path):
    # Instanciate GPT2 language model
    gpt2_config = tm.GPT2Config(vocab_size, params["seqlen"], params["n_ctx"], params["embed"], params["layers"], params["heads"],
                               resid_pdrop=params["drop"], embd_pdrop=params["drop"], attn_pdrop=params["drop"])

    # Load GPT2 language model trained weights
    language_model = GPT2LanguadeModel(gpt2_config)
    ckpt = tf.train.Checkpoint(net=language_model)
    ckpt.restore(tf.train.latest_checkpoint(path))

    return language_model

def load_clf_dnd(vocab_size, path):
    # Instanciate Bert text emotion classifier
    bert_config = tm.BertConfig(vocab_size, hidden_size=256, num_hidden_layers=2, num_attention_heads=8, num_labels=4)

    # Load Bert trained weights
    clf_dnd = Bert(bert_config)
    ckpt = tf.train.Checkpoint(net=clf_dnd)
    ckpt.restore(tf.train.latest_checkpoint(path))

    return clf_dnd

def load_clf_vgmidi(vocab_size, params, path="../trained/clf_vgmidi.ckpt"):
    # Instanciate GPT2 music emotion classifier
    gpt2_config = tm.GPT2Config(vocab_size, params["seqlen"], params["n_ctx"], params["embed"], params["layers"], params["heads"],
                               resid_pdrop=params["drop"], embd_pdrop=params["drop"], attn_pdrop=params["drop"])

    # Load pre-trained GPT2 without language model head
    clf_vgmidi = GPT2Classifier(gpt2_config)
    ckpt = tf.train.Checkpoint(net=clf_vgmidi)
    ckpt.restore(tf.train.latest_checkpoint(path))

    return clf_vgmidi

def generate_midi(generation_params, language_model, clf_vgmidi, clf_dnd):
    init_tokens = generation_params["init_tokens"]

    generated = np.array([init_tokens])

    while generated.shape[-1] < gen_len:
        # Run language_model
        generative_x = generated[:, -n_ctx:]
        generative_y = language_model(generative_x, training=False)
        generative_p = tf.math.softmax(generative_y)

        # Get topk most likely tokens from the language model
        top_probs, top_tokens = tf.math.top_k(generative_p, top_k)

        # Create tensor with all possible tokens
        top_tokens = np.reshape(top_tokens, (top_tokens.shape[-1], 1))

        # Concatenate sequence of tokens generated so far with all possible tokens
        generated_tiled = np.tile(generated, top_tokens.shape)

        # Classifiy emotion considering all possible tokens
        music_x = np.concatenate((generated_tiled, top_tokens), axis=1)
        music_y = clf_vgmidi(music_x, training=False)
        music_p = tf.math.sigmoid(music_y)

    return generated

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
    clf_vgmidi = load_clf_vgmidi(vocab_size, params, "../trained/clf_gpt2.ckpt/clf_gpt2")

    # Load generative language model
    clf_dnd = load_clf_dnd(vocab_size, "../trained/clf_bert.ckpt/clf_bert_ep1.txt/clf_bert")

    # Encode init text as sequence of indices
    init_music = preprocess_text(opt.init)
    init_tokens = [vocab[word] for word in init_music.split(" ")]

    # Compute emotion in the given story using dnd classifier
    story_tokens = tokenizer.encode(opt.text, add_special_tokens=True)
    story_emotion = clf_dnd(story_tokens)
    print(story_emotion)

    # Generation parameters
    generation_params = {"init_tokens": init_tokens,
                              "length": opt.glen,
                                "topk": opt.topk,
                                "sent": story_emotion}

    # Generate a midi as text
    midi_text = generate_midi(generation_params, language_model, clf_vgmidi, clf_dnd)
    print(midi_text)
