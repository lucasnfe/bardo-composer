import numpy as np
import tensorflow as tf
import clf_vgmidi.midi.encoder as me

from gnt_utils import *

def baseline(generation_params, idx2token):
    story_emotion = generation_params["emotion"]
    vgmidi = generation_params["vgmidi"]

    print(vgmidi)
