import tensorflow as tf
from clf_vgmidi.models import *

def load_language_model(vocab_size, params, path):
    # Instanciate GPT2 language model
    gpt2_config = tm.GPT2Config(vocab_size, params["seqlen"], params["n_ctx"], params["embed"], params["layers"], params["heads"],
                               resid_pdrop=params["drop"], embd_pdrop=params["drop"], attn_pdrop=params["drop"])

    # Load GPT2 language model trained weights
    language_model = GPT2LanguadeModel(gpt2_config)
    ckpt = tf.train.Checkpoint(net=language_model)
    ckpt.restore(tf.train.latest_checkpoint(path))

    return language_model

def load_clf_dnd(path):
    # Instanciate Bert text emotion classifier
    bert_config = tm.BertConfig(len(vocabulary), hidden_size=256, num_hidden_layers=2, num_attention_heads=8, num_labels=4)

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
    clf_vgmidi = GPT2Classifier(clf_config)
    ckpt = tf.train.Checkpoint(net=clf_vgmidi)
    ckpt.restore(tf.train.latest_checkpoint(path))

    return clf_vgmidi

if __name__ == "__main__":
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser(description='midi_generator.py')
    parser.add_argument('--conf', type=str, required=True, help="JSON file with training parameters.")
    parser.add_argument('--mode', type=str, default="sample", help="Generation strategy.")
    parser.add_argument('--init', type=str, required=False, help="Seed text to start generation.")
    parser.add_argument('--glen', type=int, default=256, help="Length of generated midi.")
    parser.add_argument('--topk', type=int, default=10, help="Top k tokens to consider when sampling.")
    parser.add_argument('--emo', type=int, required=False, help="Sentiment of the desired midi.")

    opt = parser.parse_args()

    # Load training parameters
    params = {}
    with open(opt.conf) as conf_file:
        params = json.load(conf_file)["clf_gpt2"]

    # Load char2idx dict from json file
    with open(params["vocab"]) as f:
        vocab = json.load(f)

    # Calculate vocab_size from char2idx dict
    vocab_size = len(vocab)

    # Load generative language model
    language_model = load_language_model(vocab_size, params, "../trained/transformer.ckpt")

    # Load generative language model
    clf_vgmidi = load_clf_vgmidi(vocab_size, params, "../trained/clf_vgmidi.ckpt/clf_vgmidi")

    # Load generative language model
    clf_dnd = load_clf_dnd("../trained/clf_dnd.ckpt/clf_dnd")
