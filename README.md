# CarND-05-Deep-Learning-Deep-Neural-Networks-Basic
  Udacity Self-Driving Car Engineer Nanodegree: Deep Neural Network

## ReLu (Rectified linear unit)

A Rectified linear unit (ReLU) is type of activation function that is defined as ``f(x) = max(0, x)``.

<img src="https://github.com/ChenBohan/AI-ML-DL-03-Intro-to-Deep-Neural-Network/blob/master/readme_img/ReLus.png" width = "50%" height = "50%" div align=center />

<img src="https://github.com/ChenBohan/AI-ML-DL-03-Intro-to-Deep-Neural-Network/blob/master/readme_img/backprop.png" width = "50%" height = "50%" div align=center />

```python
hidden_layer = tf.add(tf.matmul(features, weights[0]), biases[0])
hidden_layer = tf.nn.relu(hidden_layer)
logits = tf.add(tf.matmul(hidden_layer, weights[1]), biases[1])
```

### Optimizer

```python
# Define loss and optimizer
cost = tf.reduce_mean(\
    tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\
    .minimize(cost)
```

### Session

```python
# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
```

Calling the ``mnist.train.next_batch()`` function returns a subset of the training data. 

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
