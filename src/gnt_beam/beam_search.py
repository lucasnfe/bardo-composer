import numpy as np
import tensorflow as tf

from gnt_utils import *

class BeamNode:
    def __init__(self, tokens, gen_ps):
        self._tokens = tokens
        self._logps = gen_ps

    def __str__(self):
        return str(self.tokens().squeeze())

    def tokens(self):
        return self._tokens.numpy();

    def gen_ps(self):
        return self._logps.numpy();

    def get_top_gen_sequence(self):
        max_sent = tf.argmax(self._logps)
        return list(self.tokens()[max_sent])

    def forward(self, generation_params, language_model, clf_vgmidi_valence, clf_vgmidi_arousal, tokenizer):
        story_emotion = generation_params["emotion"]
        init_tokens   = generation_params["init_tokens"]
        gen_len       = generation_params["length"]
        vocab_size    = generation_params["vocab_size"]
        n_ctx         = generation_params["n_ctx"]
        top_k         = generation_params["top_k"]
        beam_width    = generation_params["beam_width"]

        # Get probabilities of next token
        token_p = run_language_model(self._tokens, language_model, n_ctx)

        # Get the top_k token candidates
        top_k_probs, top_k_tokens = tf.math.top_k(token_p, top_k)
        top_k_probs = np.reshape(top_k_probs, [-1])
        top_k_tokens = np.reshape(top_k_tokens, [top_k * beam_width, 1])

        # Batchfy music
        repeats = [top_k for i in range(beam_width)]

        tokens = tf.repeat(self._tokens, repeats=repeats, axis=0)
        logps = tf.repeat(self._logps, repeats=repeats, axis=0).numpy().squeeze()

        music_x = tf.concat((tokens, top_k_tokens), axis=1)

        # Get probabilities of next tokens being of the story emotion
        music_valence, music_arousal = classify_music_emotion(music_x[:,-n_ctx:], story_emotion, clf_vgmidi_valence, clf_vgmidi_arousal)

        # Compute final log probability
        final_logp = logps + np.log(top_k_probs) + np.log(music_valence) + np.log(music_arousal)

        # Select top_k tokens to form first beam
        #top_k_probs, top_k_tokens = tf.math.top_k(final_logp, top_k)
        beam_tokens = sample_without_replacement(final_logp, beam_width)
        #beam_probs = final_logp[beam_tokens]
        beam_probs = (np.log(music_valence) + np.log(music_arousal))[beam_tokens]

        # Reshape init_tokens and top_k_probs to be of shape (beam_width, 1)
        beam_tokens = tf.reshape(beam_tokens, (beam_width, 1))
        beam_probs = tf.reshape(beam_probs, (beam_width, 1))

        curt_beam = tf.gather_nd(music_x, beam_tokens)

        # Create first beam step
        c_node = BeamNode(curt_beam, beam_probs)

        return c_node

def beam_search(generation_params, language_model, clf_vgmidi_valence, clf_vgmidi_arousal, tokenizer):
    n_ctx         = generation_params["n_ctx"]
    top_k         = generation_params["top_k"]
    beam_width    = generation_params["beam_width"]
    gen_len       = generation_params["length"]
    vocab_size    = generation_params["vocab_size"]
    init_tokens   = generation_params["init_tokens"]
    story_emotion = generation_params["emotion"]

    # Batchfy tokens: [beam_width, 1]
    init_tokens = tf.expand_dims(init_tokens, axis=0)

    # Get probabilities of next token
    token_p = run_language_model(init_tokens, language_model, n_ctx)

    # Get the top_k token candidates
    top_k_probs, top_k_tokens = tf.math.top_k(token_p, top_k)
    top_k_tokens = tf.reshape(top_k_tokens, [top_k, 1])

    # Batchfy music
    init_tokens = tf.tile(init_tokens, tf.constant([top_k, 1], tf.int32))
    init_tokens = tf.concat((init_tokens, top_k_tokens), axis=1)

    # Get probabilities of next tokens being of the story emotion
    music_valence, music_arousal = classify_music_emotion(init_tokens[:,-n_ctx:], story_emotion, clf_vgmidi_valence, clf_vgmidi_arousal)

    # Compute final log probability
    top_k_probs = top_k_probs.numpy().squeeze()
    top_k_tokens = top_k_tokens.numpy().squeeze()
    final_logp = np.log(top_k_probs) + np.log(music_valence) + np.log(music_arousal)

    # Select top_k tokens to form first beam
    beam_tokens = sample_without_replacement(final_logp, beam_width)
    #beam_probs = final_logp[beam_tokens]
    beam_probs = (np.log(music_valence) + np.log(music_arousal))[beam_tokens]

    # Reshape init_tokens and top_k_probs to be of shape (beam_width, 1)
    beam_tokens = tf.reshape(beam_tokens, (beam_width, 1))
    beam_probs = tf.reshape(beam_probs, (beam_width, 1))

    init_beam = tf.gather_nd(init_tokens, beam_tokens)

    # Create first beam step
    c_node = BeamNode(init_beam, beam_probs)

    for i in range(gen_len):
        # Iterate on the list of adjacent nodes
        c_node = c_node.forward(generation_params, language_model, clf_vgmidi_valence, clf_vgmidi_arousal, tokenizer)
        #print(c_node)
        # print(c_node.gen_ps())
        # print(c_node.sent_ps())

    #print(c_node.get_top_gen_sequence())
    return c_node.get_top_gen_sequence()
