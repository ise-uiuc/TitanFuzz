import tensorflow as tf

tf.compat.v1.disable_v2_behavior()
print("eager? ", tf.executing_eagerly())
