import tensorflow
import tensorflow as tf

print(tf.__version__)
gpu_devices = tf.config.list_physical_devices("GPU")
print(gpu_devices)
assert len(gpu_devices) >= 1
import torch

print(torch.__version__)

# Test if it is not allowed to overwrite source code.
try:
    tf.io.write_file("mycoverage/mp_executor.py", "hello world")
    assert False
except tf.errors.PermissionDeniedError as e:
    assert True

try:
    with open("mycoverage/mp_executor.py", "w") as f:
        f.write("hello world")
    assert False
except Exception as e:
    print(e)
    assert True

# Test it is okay to write to a new file
try:
    tf.io.write_file("output.txt", "hello world")
    assert True
except tf.errors.PermissionDeniedError as e:
    assert False

# Test it is okay to write to a new file in Results
try:
    tf.io.write_file("Results/mytest.txt", "hello world")
    assert True
except tf.errors.PermissionDeniedError as e:
    assert False

print(tf.add(3, 4))

# Loop forever to test if GPU memory is set to dynamic growth
# while True: pass
