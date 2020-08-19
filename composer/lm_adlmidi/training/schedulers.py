import math
import tensorflow as tf

class GPTSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, learning_rate, n_training_steps, schedule, warmup=0.002):
        self.learning_rate = learning_rate
        self.n_training_steps = float(n_training_steps)
        self.warmup = warmup
        self.schedule = None

        if schedule == 'warmup_constant':
            self.schedule = self.warmup_constant
        elif schedule == 'warmup_linear':
            self.schedule = self.warmup_linear
        elif schedule == 'warmup_cosine':
            self.schedule = self.warmup_cosine

    def __call__(self, step):
        return self.learning_rate * self.schedule(step/self.n_training_steps)

    def warmup_cosine(self, x):
        s = tf.cast(x <= self.warmup, tf.float32)
        return s * (x / self.warmup) + (1 - s) * (0.5 * (1 + tf.cos(math.pi * x)))

    def warmup_constant(self, x):
        s = tf.cast(x <= self.warmup, tf.float32)
        return s * (x / self.warmup) + (1 - s) * 1

    def warmup_linear(self, x):
        s = tf.cast(x <= self.warmup, tf.float32)
        return (s * (x / self.warmup) + (1 - s)) * (1 - x)

    def get_config(self):
        return {
            "learning_rate": self.learning_rate,
            "n_training_steps": self.n_training_steps,
            "warmup": self.warmup,
            "schedule": self.schedule
        }
