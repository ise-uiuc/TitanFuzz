import numpy as np
import tensorflow as tf

tf.random.set_seed(0)
layer = tf.keras.layers.Dropout(0.2, input_shape=(2,))
data = np.arange(10).reshape(5, 2).astype(np.float32)
print(data)
