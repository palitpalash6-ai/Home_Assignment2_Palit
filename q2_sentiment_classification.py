"""
Question 2: Sentiment Classification Using RNN
CS5720 - Home Assignment 2
 
Trains an LSTM-based sentiment classifier on the IMDB movie review
dataset, then evaluates it with a confusion matrix and classification
report.
"""
 
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
 
# 1. Load the IMDB sentiment dataset
# Using tensorflow.keras.datasets.imdb as specified in the assignment.
VOCAB_SIZE = 10000
MAX_LEN = 200
 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=VOCAB_SIZE)
 
print("Training samples:", len(x_train))
print("Test samples:", len(x_test))
 
# 2. Preprocess: tokenization is already done by keras (reviews are
#    already integer-encoded word sequences); we just need to pad them
#    to a fixed length so they can be batched.
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=MAX_LEN)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=MAX_LEN)
 
# 3. Build and train an LSTM-based sentiment classifier
model = tf.keras.Sequential([
    tf.keras.Input(shape=(MAX_LEN,)),
    tf.keras.layers.Embedding(VOCAB_SIZE, 128),
    tf.keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
 
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()
 
history = model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=5,
    batch_size=128,
    verbose=2,
)
 
# Evaluate on the test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")
 
# 4. Generate a confusion matrix and classification report
y_pred_probs = model.predict(x_test, verbose=0)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()
 
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
 
report = classification_report(y_test, y_pred, target_names=["Negative", "Positive"])
print("\nClassification Report:")
print(report)
 
# Plot confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - IMDB Sentiment Classification")
plt.tight_layout()
plt.savefig("imdb_confusion_matrix.png", dpi=150)
print("\nSaved imdb_confusion_matrix.png")
