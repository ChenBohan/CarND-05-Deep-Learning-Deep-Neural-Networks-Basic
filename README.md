# AI-ML-DL-03-Intro-to-Deep-Neural-Network
  Udacity Self-Driving Car Engineer Nanodegree: Deep Neural Network

## Rectified linear unit (ReLu)

<img src="https://github.com/ChenBohan/AI-ML-DL-03-Intro-to-Deep-Neural-Network/blob/master/readme_img/ReLus.png" width = "50%" height = "50%" div align=center />

<img src="https://github.com/ChenBohan/AI-ML-DL-03-Intro-to-Deep-Neural-Network/blob/master/readme_img/backprop.png" width = "50%" height = "50%" div align=center />

```python
# Weights and biases
weights = [
    tf.Variable(hidden_layer_weights),
    tf.Variable(out_weights)]
biases = [
    tf.Variable(tf.zeros(3)),
    tf.Variable(tf.zeros(2))]

# Input
features = tf.Variable([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0], [11.0, 12.0, 13.0, 14.0]])

# TODO: Create Model
hidden_layer = tf.add(tf.matmul(features, weights[0]), biases[0])
hidden_layer = tf.nn.relu(hidden_layer)
logits = tf.add(tf.matmul(hidden_layer, weights[1]), biases[1])

# TODO: save and print session results on variable output
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    output = sess.run(logits)
    print(output)
```

A Rectified linear unit (ReLU) is type of activation function that is defined as ``f(x) = max(0, x)``. 

The function returns 0 if ``x`` is negative, otherwise it returns ``x``. TensorFlow provides the ReLU function as ``tf.nn.relu()``.

## Prevent over fitting

### The first way: Early Termination

<img src="https://github.com/ChenBohan/AI-ML-DL-03-Intro-to-Deep-Neural-Network/blob/master/readme_img/Early%20Termination.png" width = "50%" height = "50%" div align=center />

### The second way: Regularization

<img src="https://github.com/ChenBohan/AI-ML-DL-03-Intro-to-Deep-Neural-Network/blob/master/readme_img/Regularization.png" width = "50%" height = "50%" div align=center />

#### Dropout

Dropout is a regularization technique for reducing overfitting.

<img src="https://github.com/ChenBohan/AI-ML-DL-03-Intro-to-Deep-Neural-Network/blob/master/readme_img/Dropout.png" width = "50%" height = "50%" div align=center />

TensorFlow provides the ``tf.nn.dropout()`` function.

```python
keep_prob = tf.placeholder(tf.float32) # probability to keep units

hidden_layer = tf.add(tf.matmul(features, weights[0]), biases[0])
hidden_layer = tf.nn.relu(hidden_layer)
hidden_layer = tf.nn.dropout(hidden_layer, keep_prob)

logits = tf.add(tf.matmul(hidden_layer, weights[1]), biases[1])
```

The tf.nn.dropout() function takes in two parameters:

1. ``hidden_layer``: the tensor to which you would like to apply dropout

2. ``keep_prob``: the probability of keeping (i.e. not dropping) any given unit

PS:

1. During training, a good starting value for keep_prob is 0.5.

2. During testing, use a keep_prob value of 1.0 to keep all units and maximize the power of the model.
