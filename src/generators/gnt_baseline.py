import numpy as np
import tensorflow as tf
import clf_vgmidi.midi.encoder as me

from gnt_utils import *

def beam_search(generation_params, idx2token):
    
