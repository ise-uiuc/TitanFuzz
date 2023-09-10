# README

## Set-up docker
1. upload seed programs to codex_seed_programs/
```
codex_seed_programs/
    codex_tf_0.4_256_25/
    codex_torch_0.4_256_25/
lmfuzz/
    scripts/
    ...
```

2. Run a new container from docker image `lmfuzz:latest`, by default the container will be named lmfuzz
```
bash scripts/docker/initdocker.sh
```

3. Run the following command to copy the current directory to docker after code changes.
```
bash copyfile.sh
```
Check all permission works.
```
bash scripts/dockerrun.sh bash scripts/docker/testdocker.sh
```

4. Run experiments by running scripts
```
bash dockerrun.sh bash gv_tf_script.sh
```
or interactively in docker
```
bash dockerrun.sh /bin/bash
```

## TensorFlow environment setting

There are a few environment setting to pay attention to when running experiments in TensorFlow (If run in docker, these problems should automatically be resolved with `dockerrun.sh`):

1. On boba tensorflow cannot automatically locate cuda, we need to set environment variable like this:

```
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda/
```

2. Limit GPU memory growth [guide](https://www.tensorflow.org/guide/gpu). Otherwise TF will allocate all GPU memory from start.

We implement the `tf.config.experimental.set_memory_growth` solution in [set_memory_growth](https://github.com/brutalsavage/lmfuzz/blob/main/util/util.py#L371), but sometimes it's not enough, we need to set the environment variable like this:
```
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

3. Disable annoying warnings (optional)

After importing tensorflow, there are many lines of warnings like this
```
Successful NUMA node read from SysFS had negative value (-1)
```
See [issue](https://github.com/tensorflow/tensorflow/issues/42738). These messages don't indicate an actually error. To test if GPU is available, use:
```
tf.test.is_gpu_available()
```

To get rid of the warnings, we can use these lines:
```
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
```
or
```
export TF_CPP_MIN_LOG_LEVEL='3'
```

## Oracle usage

For all following examples, add `--tf` to the arguments list if you are running tensorflow. The oracle uses torch by default.

- Generate CPU/GPU comparison results for a batch of code

```
python driver.py --mode race --input ./folder_containing_all_code_files
```

- Compare CPU/GPU results on a single testcase

```
python torch2cuda.py --mode dual --input ./testcase.py
```

- Try to find the first line of code where CPU/GPU begin to show inconsistency

```
python torch2cuda.py --mode duel --input ./testcase.py
```

- Run a testcase with CPU passes and print the transformed code

```
python torch2cuda.py --mode single --input ./testcase.py --cpu --code
```

- Run a testcase with GPU passes and print the transformed code

```
python torch2cuda.py --mode single --input ./testcase.py --code
```
