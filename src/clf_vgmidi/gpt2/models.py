import tensorflow as tf
import transformers.modeling_tf_gpt2 as tm

class GPT2Classifier(tm.TFGPT2Model):
    def __init__(self, config, num_labels=1, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # self.dropout = tf.keras.layers.Dropout(0.5)
        self.emotion_head = tf.keras.layers.Dense(num_labels)

    def call(self, inputs, **kwargs):
        # Extract features
        outputs = super().call(inputs, **kwargs)

        #  Extract language model logits
        lm_logits = self.transformer.wte(outputs[0], mode="linear", training=kwargs["training"])

        # Finetuner Emotion Head
        emotion_logits = self.emotion_head(outputs[0], training=kwargs["training"])

        return (emotion_logits, lm_logits)
