import training.load_data
import training.schedulers
import training.checkpoint
import tensorflow as tf

def generative_loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

def perplexity(labels, logits):
    """
    Popular metric for evaluating language modelling architectures.
    More info: http://cs224d.stanford.edu/lecture_notes/LectureNotes4.pdf
    """
    cross_entropy = generative_loss(labels, logits)
    return tf.keras.backend.mean(tf.keras.backend.exp(tf.keras.backend.mean(cross_entropy, axis=-1)))

def calc_steps(dataset_midi_files, seq_length, batch_size):
    # Get a list of the txt files associated to the midi files
    dataset_txt_files = training.load_data.midi2text_paths(dataset_midi_files)

    # Read all files in the dataset directory
    list_ds = tf.data.Dataset.from_tensor_slices(dataset_txt_files)

    n_steps = 0
    for filepath in list_ds:
        text = tf.io.read_file(filepath)
        words = tf.strings.split(text, sep=" ")

        n_tokens = words.shape[-1]
        n_chunks = n_tokens//(seq_length + 1)

        n_steps += n_chunks//batch_size

    return n_steps

def train_language_model(language_model, params, train_dataset, test_dataset, n_train_steps):
    # Compile model with given optimizer and defined loss
    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # with mirrored_strategy.scope():
    lr_schedule = training.schedulers.GPTSchedule(learning_rate=params["lr"],
                                      n_training_steps=n_train_steps * params["epochs"],
                                              schedule=params["schedule"],
                                                warmup=params["warmup"])

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    language_model.compile(optimizer, loss=generative_loss, metrics=[perplexity])

    # Add checkpoint callback
    weights_callback = training.checkpoint.SaveModelCallback(language_model, optimizer, params["check"])
    history = language_model.fit(train_dataset, epochs=params["epochs"], validation_data=test_dataset, callbacks=[weights_callback])

    return history
