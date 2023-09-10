import tensorflow as tf

shape = [500, 500]
x = tf.random.uniform(shape)
y = tf.random.normal(shape)
print(tf.matmul(x, y))
