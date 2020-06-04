import tensorflow as tf
import transformers as tm

class GPT2LanguadeModel(tm.modeling_tf_gpt2.TFGPT2Model):
    def call(self, inputs, **kwargs):
        gpt_outputs = super().call(inputs, **kwargs)
        lm_logits = self.transformer.wte(gpt_outputs[0], mode="linear")

        return lm_logits[:,-1,:]

class GPT2Classifier(tm.modeling_tf_gpt2.TFGPT2Model):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.emotion_head = tf.keras.layers.Dense(1, name="emotion_head")

    def call(self, inputs, **kwargs):
        gpt_outputs = super().call(inputs, **kwargs)
        emotion_logits = self.emotion_head(gpt_outputs[0][:,-1,:])
        return emotion_logits
