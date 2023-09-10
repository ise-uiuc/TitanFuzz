import tensorflow as tf

if not tf.executing_eagerly():
    raise Exception("Not eager!")
