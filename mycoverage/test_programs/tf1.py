import tensorflow as tf

# The inputs are 28x28 RGB images with `channels_last` and the batch
# size is 4.
input_shape = (4, 28, 28, 3)
x = tf.random.normal(input_shape)
print(x.numpy().shape)
y = tf.keras.layers.Conv2D(2, 3, activation="relu", input_shape=input_shape[1:])(x)
print(y.shape)
