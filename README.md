# Keras
It is a high-level deep learning API written in Python by Google, designed to make building and experimenting with neural networks fast, easy, and intuitive. It acts as an interface for the TensorFlow library, but it can also run on top of other backends like Theano and Microsoft Cognitive Toolkit (CNTK), though TensorFlow is now the default and most supported backend.
**Key Features of Keras**
- User-friendly: Simple API that minimizes the number of user actions required for common use cases.
- Modular: Neural networks are built by plugging together building blocks like layers, optimizers, and activation functions.
- Extensible: Easily add new modules or custom components.
- Pythonic: Follows Python conventions and practices, making it readable and easy to debug.
- Integrated with TensorFlow: Keras is now part of TensorFlow and accessible via tf.keras.
**Core Components of Keras**
1. Models

Keras provides two main ways to define a model:
- Sequential – Linear stack of layers.
- Model (Functional API) – More flexible, supports complex architectures like multi-input/output.
2. Layers

Each Layer in Keras is like a transformation function. For example:
- Dense(units, activation): Fully connected layer.
- Conv2D(filters, kernel_size): Used in image processing.
- LSTM(units): Used for sequence data like time series or text.

Each layer has:
- weights (learned during training)
- activation function (adds non-linearity)
- input/output shape

3. Activation Functions
They define how the weighted sum of the input is transformed before being passed to the next layer.
- relu: For hidden layers (fast and efficient).
- sigmoid: For binary classification.
- softmax: For multi-class classification.
- tanh: Alternative to sigmoid, but zero-centered.

4. Loss Functions

Used to measure the difference between actual and predicted values.
- binary_crossentropy: For binary classification.
- ategorical_crossentropy: For multi-class (one-hot encoded).
- sparse_categorical_crossentropy: For multi-class (integers as labels).
- mse (mean squared error): For regression tasks.

5. Optimizers

These define how the model weights are updated based on the loss.
- Adam: Adaptive learning rate; works well in most cases.
- SGD: Stochastic gradient descent (basic and fast).
- RMSprop: Useful in RNNs and sequence models.

5. Metrics

Used to evaluate performance.
- accuracy: Classification.
- mae: Mean Absolute Error (regression).
- mse: Mean Squared Error.
*Use Cases of Keras*
- Image classification
- Natural language processing
- Time series forecasting
- Recommender systems
- Transfer learning
**Basic Keras Workflow**
```python 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define a simple model
model = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=(10,)),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Dummy data
import numpy as np
X_train = np.random.rand(1000, 10)
y_train = np.random.randint(2, size=(1000, 1))

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)
```

**Example: Binary Classification of Breast Cancer**

We’ll use the Breast Cancer Wisconsin dataset (available in sklearn) to classify tumors as benign or malignant.

```python 
1. Load Libraries and Dataset
# import Libraries
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
```

2. Prepare the Data

```python 
# Load Dataset
data = load_breast_cancer()

# print(data.shape)
# data.head()

X = data.data
y = data.target
```

```python
# Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=80)   
```

```python
# Scaling the data  
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

3. Define and Train a Simple Neural Network

```python
X_train.shape
# 
X_train.shape[1]
```

```python
# building the model
model = Sequential([
    Dense(30, activation = 'relu', input_shape=(X_train.shape[1], )), # Input layer with number of features
    Dense(15, activation = 'relu'), # Neurons in hidden layer
    Dense(1, activation = 'sigmoid') # Output layer for binary classification
])
```

```python
# compile the model
model.compile(
    optimizer = Adam(learning_rate = 0.001),
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)
```

```python
# fit the model 
model.fit(
    X_train, y_train, epochs = 15, batch_size = 32,
    validation_split = 0.2
)
```

```python
# evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f'test accuracy:  {accuracy:.4f}')
```

# CNNs and RNNs
Here’s a curated list of popular pretrained CNNs and RNNs, where they're typically used, and how you can access them (especially via TensorFlow or PyTorch).
**Pretrained CNNs**

These are trained on ImageNet (1.2 million labeled images across 1,000 categories) and are commonly used for image classification, object detection, segmentation, etc.
| Model            | Size / Speed   | Accuracy  | Use Case                                     |
| ---------------- | -------------- | --------- | -------------------------------------------- |
| **VGG16/VGG19**  | Large, slow    | Moderate  | Simple structure, good for transfer learning |
| **ResNet50/101** | Medium         | High      | Very deep network, great generalization      |
| **MobileNetV2**  | Lightweight    | Good      | Mobile/web apps, low-resource devices        |
| **EfficientNet** | Compact + fast | Very high | Best balance of speed & accuracy             |
| **InceptionV3**  | Medium         | High      | Complex objects, fine-grained tasks          |
| **DenseNet121**  | Small-medium   | High      | Feature reuse, strong performance           
Using a pretrained CNN (TensorFlow / Keras)
from tensorflow.keras.applications import ResNet50

```python
model = ResNet50(weights='imagenet')  # Downloaded automatically
```
OR

```python
from tensorflow.keras.applications import (
    VGG16, VGG19, InceptionV3, MobileNetV2, DenseNet121, EfficientNetB0
)
```

**Pretrained RNNs**

Unlike CNNs, pretrained RNNs are often embedded in NLP-focused models rather than standalone. They’re used for tasks like sentiment analysis, translation, speech recognition, and time series prediction.

| Model               | Type            | Use Case                                   | Notes                                    |
| ------------------- | --------------- | ------------------------------------------ | ---------------------------------------- |
| **Word2Vec + LSTM** | RNN + Embedding | Sentiment analysis, text classification    | Pretrained embeddings + RNN              |
| **ELMo**            | 2-layer Bi-LSTM | Contextual word embeddings (NLP)           | Deep pretrained LSTM encoder             |
| **Seq2Seq**         | LSTM/GRU        | Machine translation, text summarization    | Often custom-trained on translation data |
| **Tacotron 2**      | LSTM + CNN      | Text-to-speech generation                  | Uses CNNs + RNNs                         |
| **OpenNMT**         | LSTM            | Neural machine translation (multi-lingual) | PyTorch / TensorFlow                     |

**RNNs are now often replaced by Transformers**

In modern NLP, RNNs are being replaced by Transformers like:

| Transformer Model | Replaces RNN in | Pretrained? | Examples                   |
| ----------------- | --------------- | ----------- | -------------------------- |
| **BERT**          | Text encoding   | ✅           | Question answering, NER    |
| **GPT**           | Text generation | ✅           | Chatbots, summarization    |
| **T5 / BART**     | Seq2seq tasks   | ✅           | Translation, summarization |

```python
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential

model = Sequential([
    Embedding(input_dim=10000, output_dim=300, input_length=100),  # Could load GloVe here
    LSTM(64),
    Dense(1, activation='sigmoid')
])
```

**Summary Table**
| Type  | Model Name    | Use Case                    | Pretrained in                 |
| ----- | ------------- | --------------------------- | ----------------------------- |
| CNN   | ResNet50      | Image classification        | `keras.applications`          |
| CNN   | EfficientNet  | Mobile/web vision tasks     | `keras.applications`          |
| RNN   | ELMo          | NLP tasks (contextual word) | `tensorflow-hub`              |
| RNN   | Seq2Seq LSTM  | Translation, summarization  | `OpenNMT`, `Fairseq`          |
| Combo | Tacotron 2    | Text-to-speech              | `NVIDIA DeepLearningExamples` |
| NLP   | BERT, GPT, T5 | Replaces RNN in many tasks  | `HuggingFace Transformers`    |