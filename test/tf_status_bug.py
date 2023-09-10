import tensorflow as tf

data = ["a", "b", "c", "d", "e"]
partitions = [3, -2, 2, -1, 2]
num_partitions = 5
t1 = tf.ragged.stack_dynamic_partitions(data, partitions, num_partitions)  # Succeed
print(t1)
# t2 = tf.ragged.stack(tf.dynamic_partition(data, partitions, num_partitions)) # Raise InvalidArgumentError
# print(t2)
