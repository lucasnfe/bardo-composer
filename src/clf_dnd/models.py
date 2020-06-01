import tensorflow as tf
import transformers as tm

class Bert(tm.TFBertForSequenceClassification):
    def call(self, inputs, **kwargs):
        outputs = super().call(inputs, **kwargs)
        return outputs[0]
