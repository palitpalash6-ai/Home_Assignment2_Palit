# Home Assignment 2 — Neural Networks and Deep Learning (CS5720)

**Student Name:** Palash Palit
**Course:** CS5720 Neural Network and Deep Learning — Fall 2026
**University:** University of Central Missouri, Department of Computer Science & Cybersecurity

## Overview

This folder contains the code for Home Assignment 2, which covers recurrent
networks (RNN/LSTM) and convolutional networks (CNN). Every script is
commented to explain each step.

| File | Question | What it does |
|---|---|---|
| `q1_rnn_text_generation.py` | Q1 | Trains a character-level LSTM on Shakespeare's text and generates new text at three temperatures (0.5, 1.0, 1.5) |
| `q2_sentiment_classification.py` | Q2 | Trains an LSTM to classify IMDB movie reviews as positive or negative, then prints a confusion matrix and classification report |
| `q3_convolution.py` | Q3 | Convolves a 5×5 matrix with a 3×3 kernel using stride 1 and 2, and VALID and SAME padding |
| `q4_task1_sobel.py` | Q4 Task 1 | Applies Sobel-X and Sobel-Y filters for edge detection |
| `q4_task2_pooling.py` | Q4 Task 2 | Applies 2×2 max pooling and average pooling to a random 4×4 matrix |
| `q5_task1_alexnet.py` | Q5 Task 1 | Builds a simplified AlexNet and prints its model summary |
| `q5_task2_resnet.py` | Q5 Task 2 | Builds a residual block with a skip connection and a small ResNet |
| `shakespeare.txt` | Q1 | Tiny Shakespeare text corpus (public domain) used for training in Q1 |
| `sobel_edge_detection.png` | Q4 | Output image from the Sobel filters |
| `imdb_confusion_matrix.png` | Q2 | Confusion matrix from the sentiment classifier |
| `HomeAssignment2_Colab.ipynb` | All | Google Colab notebook with every question and its actual outputs |

## How to Run

Requires Python 3.9–3.12 (TensorFlow does not support newer versions yet).

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn opencv-python
```

Run each script from this folder:

```bash
python q1_rnn_text_generation.py
python q2_sentiment_classification.py
python q3_convolution.py
python q4_task1_sobel.py
python q4_task2_pooling.py
python q5_task1_alexnet.py
python q5_task2_resnet.py
```

Q2 downloads the IMDB dataset automatically the first time, so it needs
internet access. Q1 and Q2 train neural networks and can take several minutes
on a CPU. Google Colab with a GPU is faster; `HomeAssignment2_Colab.ipynb` runs everything in one go.

## Summary of Each Question

### Q1 — RNN Text Generation
The text is converted to a sequence of characters, and an LSTM learns to
predict the next character. After training, the model generates new text one
character at a time.

**Temperature scaling** divides the model's output scores before sampling:
- Low temperature (0.5) gives safer, more repetitive text.
- High temperature (1.5) gives more varied but less coherent text.

In my run (5 epochs on the full corpus, Google Colab T4 GPU), the loss dropped from 2.46 to 1.54, and the generated text shows speaker names, line breaks, and mostly real words.

### Q2 — Sentiment Classification
Reviews are padded to a fixed length and passed through an Embedding layer,
an LSTM, and a sigmoid output. The confusion matrix and classification report
show how well the model handles each class separately.

**Result (full 25,000/25,000 IMDB dataset):** 84.6% test accuracy.

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Negative | 0.86 | 0.83 | 0.84 |
| Positive | 0.84 | 0.86 | 0.85 |

**Why precision and recall matter:** accuracy alone doesn't show which class
the errors fall in. Here they are fairly balanced, but the model labels
negative reviews as positive (2,096) a bit more often than the reverse (1,760),
so Negative recall is slightly lower. Which metric matters more depends on the
application.

### Q3 — Convolution with Different Parameters
The same kernel is applied four ways:

| Stride | Padding | Output shape |
|---|---|---|
| 1 | VALID | 3 × 3 |
| 1 | SAME | 5 × 5 |
| 2 | VALID | 2 × 2 |
| 2 | SAME | 3 × 3 |

The kernel is a Laplacian (second-derivative) filter, and the input increases
evenly, so the interior outputs are 0. Non-zero values appear only at the
padded borders.

### Q4 — Sobel Filters and Pooling
Sobel-X detects vertical edges and Sobel-Y detects horizontal edges. Max
pooling keeps the largest value in each 2×2 block; average pooling keeps the
mean.

### Q5 — AlexNet and ResNet
AlexNet stacks five convolutional layers and three dense layers
(about 58 million parameters). The ResNet residual block adds its input to
its output through a skip connection, which lets gradients flow through deep
networks.
