import tensorflow as tf
import transformers.modeling_tf_gpt2 as tm

class GPT2LanguageModel(tm.TFGPT2LMHeadModel):
    def call(self, inputs, **kwargs):
        outputs = super().call(inputs, **kwargs)
        return outputs[0]

class GPT2Classifier(tm.TFGPT2Model):
    def __init__(self, config, num_labels=1, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # self.dropout = tf.keras.layers.Dropout(0.5)
        self.emotion_head = tf.keras.layers.Dense(num_labels)

    def call(self, inputs, **kwargs):
        # Extract features
        outputs = super().call(inputs, **kwargs)

        # Finetuner Emotion Head
        emotion_logits = self.emotion_head(outputs[0], training=kwargs["training"])

        return (emotion_logits, lm_logits)
