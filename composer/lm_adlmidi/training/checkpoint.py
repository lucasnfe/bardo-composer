import os
import json
import numpy as np
import tensorflow as tf

class SaveModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, optimizer, save_path, max_to_keep=1):
        self.save_path = save_path

        # Define checpoint and manager
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
        self.manager = tf.train.CheckpointManager(self.ckpt, save_path, max_to_keep=max_to_keep)

        # Load latest checkpoint
        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        # Load previous logs
        logs_path = os.path.join(self.save_path, "logs.json")
        if os.path.isfile(logs_path):
            with open(logs_path, "r") as logs_file:
                self.logs = json.load(logs_file)
                self.best = float(self.logs.get("val_loss"))
        else:
            self.logs = {}
            self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        self._save_model(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        self.ckpt.step.assign_add(1)

    def _save_model(self, epoch, logs):
        logs = logs or {}

        # Only save model if monitored metric has improved
        current = logs.get("val_loss")
        if current and current < self.best:
            print('\nEpoch %05d: val_loss improved from %0.5f to %0.5f,'
                  ' saving model to %s' % (epoch + 1, self.best, current, self.save_path))

            self.best = current

            # Save model and optimizer states
            self.manager.save()

            # Save logs file
            logs_path = os.path.join(self.save_path, "logs.json")
            with open(logs_path, "w") as logs_file:
                current_logs = {k:format(v, '.5f') for k,v in logs.items()}
                json.dump(current_logs, logs_file)
        else:
            print('\nEpoch %05d: val_loss did not improve from %0.5f' % (epoch + 1, self.best))
