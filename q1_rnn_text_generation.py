"""
Question 1: Implementing an RNN for Text Generation
CS5720 - Home Assignment 2
 
Trains a character-level LSTM to predict the next character in a
text dataset (Shakespeare's works), then generates new text by
sampling characters one at a time.
"""
 
import numpy as np
import tensorflow as tf
 
# 1. Load the text dataset
# (Tiny Shakespeare corpus -- a widely used public-domain text dataset
#  for character-level language modeling. Download it once, or replace
#  this path with any text file of your choice, e.g. "The Little Prince".)
with open("shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()
 
print("Total characters in corpus:", len(text))
 
# 2. Convert text into a sequence of characters (integer-encoded)
vocab = sorted(set(text))
vocab_size = len(vocab)
print("Unique characters (vocabulary size):", vocab_size)
 
char_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_char = {i: ch for i, ch in enumerate(vocab)}
 
encoded_text = np.array([char_to_idx[ch] for ch in text])
 
# Build (input_sequence -> next_character) training pairs using a sliding window
SEQ_LENGTH = 100
examples_per_epoch = len(text) // (SEQ_LENGTH + 1)
 
char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)
sequences = char_dataset.batch(SEQ_LENGTH + 1, drop_remainder=True)
 
def split_input_target(chunk):
    input_seq = chunk[:-1]
    target_seq = chunk[1:]
    return input_seq, target_seq
 
dataset = sequences.map(split_input_target)
 
BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
 
# 3. Define an RNN model using LSTM layers to predict the next character
EMBEDDING_DIM = 256
RNN_UNITS = 512
 
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.Input(batch_shape=(batch_size, None)),
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer="glorot_uniform"),
        tf.keras.layers.Dense(vocab_size),
    ])
    return model
 
model = build_model(vocab_size, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
model.summary()
 
def loss_fn(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
 
model.compile(optimizer="adam", loss=loss_fn)
 
# 4. Train the model
EPOCHS = 5
history = model.fit(dataset, epochs=EPOCHS)
 
# Rebuild the model with batch_size=1 for text generation, and copy trained weights
gen_model = build_model(vocab_size, EMBEDDING_DIM, RNN_UNITS, batch_size=1)
gen_model.set_weights(model.get_weights())
gen_model.build(tf.TensorShape([1, None]))
 
 
def generate_text(model, start_string, num_generate=300, temperature=1.0):
    input_eval = [char_to_idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
 
    text_generated = []
    for layer in model.layers:
        if hasattr(layer, "reset_states"):
            layer.reset_states()
 
    for _ in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
 
        # Apply temperature scaling before sampling
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
 
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx_to_char[predicted_id])
 
    return start_string + "".join(text_generated)
 
 
# 5. Generate text by sampling characters, at a few different temperatures
print("\n" + "="*60)
print("Generated text (temperature = 0.5, more conservative):")
print("="*60)
print(generate_text(gen_model, start_string="ROMEO: ", temperature=0.5))
 
print("\n" + "="*60)
print("Generated text (temperature = 1.0, balanced):")
print("="*60)
print(generate_text(gen_model, start_string="ROMEO: ", temperature=1.0))
 
print("\n" + "="*60)
print("Generated text (temperature = 1.5, more random):")
print("="*60)
print(generate_text(gen_model, start_string="ROMEO: ", temperature=1.5))
