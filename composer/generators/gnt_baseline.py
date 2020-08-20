import numpy as np
import tensorflow as tf
import composer.midi_encoder as me

from gnt_utils import *

def baseline(generation_params, idx2token):
    story_emotion  = generation_params["emotion"]
    vgmidi         = generation_params["vgmidi"]
    gen_len        = generation_params["length"]
    prev_piece     =  generation_params["previous"]

    # Load all pieces in the vgmidi dataset with the desired emotion
    pieces_with_story_emotion = load_vgmidi_pieces_with_emotion(vgmidi, story_emotion)

    # Check if a previous piece was given. If so, keep playing it.
    # Otherwise, select a new piece at random.
    if prev_piece:
        prev_ix, prev_token = prev_piece
    else:
        prev_token = 0
        prev_ix = np.random.randint(len(pieces_with_story_emotion))

    selected_piece, _ = pieces_with_story_emotion[prev_ix]

    # Create buffer to fill with music tokens from the selected piece
    generated_piece = []

    # Fill the buffer until the desired duration is reached
    total_duration = 0
    while total_duration <= gen_len:
        generated_piece.append(selected_piece[prev_token])

        generated_text = " ".join([idx2token[ix] for ix in generated_piece])
        total_duration = me.parse_total_duration_from_text(generated_text)

        prev_token += 1

    # The probability of the generated piece is zero beause it comes from
    # human composers
    top_sequence_prob = 0
    return generated_piece, top_sequence_prob, generated_text, (prev_ix, prev_token)
