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
        return self.tokens()[max_sent]

    def forward(self, generation_params, language_model, clf_vgmidi_valence, clf_vgmidi_arousal, tokenizer):
        story_emotion = generation_params["emotion"]
        init_tokens   = generation_params["init_tokens"]
        gen_len       = generation_params["length"]
        vocab_size    = generation_params["vocab_size"]
        n_ctx         = generation_params["n_ctx"]
        top_k         = generation_params["top_k"]

        # Get probabilities of next token
        token_p = run_language_model(self._tokens, language_model, n_ctx)
        token_p = np.reshape(token_p, [-1])
    
        # Batchfy music
        repeats = [vocab_size for i in range(self._tokens.shape[0])]
        
        tokens = tf.repeat(self._tokens, repeats=repeats, axis=0)
        logps = tf.repeat(self._logps, repeats=repeats, axis=0).numpy().squeeze()
        
        music_x = concat_all_tokens(tokens, (vocab_size, top_k), gen_len)

        # Get probabilities of next tokens being of the story emotion
        music_valence, music_arousal = classify_music_emotion(music_x, story_emotion, clf_vgmidi_valence, clf_vgmidi_arousal)

        # Compute final log probability
        final_logp = logps + tf.math.log(token_p) + tf.math.log(music_valence) + tf.math.log(music_arousal)

        # Select top_k tokens to form first beam
        top_k_probs, top_k_tokens = tf.math.top_k(final_logp, top_k)

        # Reshape init_tokens and top_k_probs to be of shape (beam_width, 1)
        top_k_tokens = tf.reshape(top_k_tokens, (top_k, 1))
        top_k_probs = tf.reshape(top_k_probs, (top_k, 1))

        init_beam = tf.gather_nd(music_x, top_k_tokens)

        # Create first beam step
        c_node = BeamNode(init_beam, top_k_probs)

        return c_node

def beam_search(generation_params, language_model, clf_vgmidi_valence, clf_vgmidi_arousal, tokenizer):
    n_ctx         = generation_params["n_ctx"]
    top_k         = generation_params["top_k"]
    gen_len       = generation_params["length"]
    vocab_size    = generation_params["vocab_size"]
    init_tokens   = generation_params["init_tokens"]
    story_emotion = generation_params["emotion"]

    # Batchfy tokens: [beam_width, 1]
    init_tokens = tf.expand_dims(init_tokens, axis=0)

    # Get probabilities of next token
    token_p = run_language_model(init_tokens, language_model, n_ctx)

    # Batchfy music
    init_tokens = tf.tile(init_tokens, tf.constant([vocab_size, 1], tf.int32))
    music_x = concat_all_tokens(init_tokens, (vocab_size, 1), gen_len)

    # Get probabilities of next tokens being of the story emotion
    music_valence, music_arousal = classify_music_emotion(music_x, story_emotion, clf_vgmidi_valence, clf_vgmidi_arousal)

    # Compute final log probability
    final_logp = tf.math.log(token_p) + tf.math.log(music_valence) + tf.math.log(music_arousal)

    # Select top_k tokens to form first beam
    top_k_probs, top_k_tokens = tf.math.top_k(final_logp, top_k)

    # Reshape init_tokens and top_k_probs to be of shape (beam_width, 1)
    top_k_tokens = tf.reshape(top_k_tokens, (top_k, 1))
    top_k_probs = tf.reshape(top_k_probs, (top_k, 1))

    init_beam = tf.gather_nd(music_x, top_k_tokens)

    # Create first beam step
    c_node = BeamNode(init_beam, top_k_probs)

    for i in range(init_tokens.shape[-1], gen_len):
        # Iterate on the list of adjacent nodes
        c_node = c_node.forward(generation_params, language_model, clf_vgmidi_valence, clf_vgmidi_arousal, tokenizer)
        # print(c_node)
        print(c_node.gen_ps())
        # print(c_node.sent_ps())

    print(c_node.get_top_gen_sequence())
    return c_node.get_top_gen_sequence()
