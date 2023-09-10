# Large Language Models Are Zero-Shot Fuzzers: Fuzzing Deep-Learning Libraries via Large Language Models
<p align="left">
    <a href="https://arxiv.org/abs/2212.14834"><img src="https://img.shields.io/badge/arXiv-2212.14834-b31b1b.svg">
    <a href="https://doi.org/10.5281/zenodo.7980923"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.7980923.svg"> </a>
</p>

This is the implementation repository of our research paper, "Large Language Models Are Zero-Shot Fuzzers: Fuzzing Deep-Learning Libraries via Large Language Models", accepted at ISSTA 2023.

## About

TitanFuzz is the first approach to directly leveraging Large Language Models (LLMs) to generate input programs for fuzzing Deep Learning (DL) libraries. LLMs are titanic models trained on billions of code snippets and can autoregressively generate human-like code snippets. Our key insight is that modern LLMs can also include numerous code snippets invoking DL library APIs in their training corpora, and thus can implicitly learn both language syntax/semantics and intricate DL API constraints for valid DL program generation. More specifically, we use both generative and infilling LLMs (e.g., Codex/InCoder) to generate and mutate valid/diverse input DL programs for fuzzing. Our experimental results demonstrate that TitanFuzz can achieve 30.38%/50.84% higher code coverage than state-of-the-art fuzzers on TensorFlow/PyTorch. Furthermore, TitanFuzz is able to detect 65 bugs, with 44 already confirmed as previously unknown bugs.

This is the TitanFuzz's implementation for fuzzing PyTorch and TensorFlow.

## Reproducibility

### Prerequisites

1. OS: A Linux System with Docker Support;

2. Hardware: GPU support.

Before you start, please make sure you have installed Docker (https://docs.docker.com/get-docker/) and nvidia-docker (https://github.com/NVIDIA/nvidia-docker).
To test if it is successfully installed, run
```
docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
```

Note: The command above and many following scripts will invoke docker commands. If your user is not in the docker group, you may encounter issues running them. To resolve this, you may either add your user to the docker group [instruction](https://docs.docker.com/engine/install/linux-postinstall/) (preferred), or run the commands with sudo `sudo docker <CMD>`.

<details><summary>Q1: Why docker?</summary>

A1: LLMs can generate arbitrary test programs, some of them that may attempt to change the source code and even cause damage to the file system. Therefore, we provide a docker environment for running TitanFuzz.

We highly recommend running `TitanFuzz` in a sandbox environment like docker. However, if you don't have docker, you may create a **conda** environment to run locally.

Below are instructions to build our conda environment `titanfuzz`:

Please run the following commands line-by-line (you may need to press `y`):
```
# Create an environment named `titanfuzz`
conda create -n titanfuzz python=3.8
# Activate
conda activate titanfuzz

# Install required packages
pip install -r requirements.txt
```

We also need to fix a bug in the library `astunparse` in order to run `TitanFuzz`. To run the following commands, you need to have `git` installed (https://github.com/git-guides/install-git).

```
# Remember to replace `/your/conda/path/` with your local conda path.
# You can check with `which python` after activating the conda environment `titanfuzz`.
cd /your/conda/path/lib/python3.8/site-packages/astunparse

# You need to have git installed to run this command.
git apply --whitespace=fix /your/local/path/titanfuzz/scripts/docker/dockerfile/unparser.patch
```
</details>


<details><summary>Q2: Why GPU?</summary>

A1: We highly recommend running LLMs on GPUs for more efficient fuzzing. In our experiments, we use a 64-core workstation with 256 GB RAM and running Ubuntu 20.04.5 LTS with 4 NVIDIA RTX A6000 GPUs. If you run on diferent GPUs and encountered OOM issues with the default generation batch size (`30`), you may consider setting a smaller batch size by changine the `BATCH_SIZE=30` line in `scripts/dockerrun.sh` and `scripts/local_run.sh`.

If you don't have GPUs, don't worry, you can still run `TitanFuzz` on CPU (although it will be much slower to generate tests). You just need to (1) set the `CUDA_DEVICE=""` in `local_run.sh` and (2) set up a local virtual environment or conda environment following the instruction in A1 (above). You can skip **Step 1** (which builds a docker environment that requires GPU) and jump to **Step 2** directly.
</details>


### Step 1 - Set up docker

<details><summary> Build the docker image </summary>

```
cd scripts/docker/dockerfile
docker build --file titanfuzz.dockerfile --tag titanfuzz:latest .
```
</details>

<details><summary> Download seed programs </summary>

We provide the codex-generated seed programs to facilitate reproduction in `TitanFuzz_codex_seed_programs.zip` ([artifact link](https://zenodo.org/record/7980923)). Suppose this repository is at `/your/local/path/titanfuzz/`, please download the seed programs to `/your/local/path/TitanFuzz_codex_seed_programs.zip`. Next, unzip it to `/your/local/path/codex_seed_programs`:

```
cd /your/local/path/
unzip TitanFuzz_codex_seed_programs.zip
```
</details>

<details><summary> Create container</summary>
Create a new docker container from docker image `titanfuzz:latest`, by default the container will be named `titanfuzz`.

```
cd titanfuzz
bash scripts/docker/initdocker.sh
bash scripts/copyfile.sh
```
</details>

### Step 2 - Run TitanFuzz

#### 1. Initial seed generation

The main script for seed generation is `codex.py`. We used Codex (code-davinci-002) for initial seed generation. However, the [Codex model](https://openai.com/blog/openai-codex) is currently unavailable, and can only be accessed via [Researcher Access Program](https://openai.com/form/researcher-access-program) or [Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/work-with-code).

For your convenience, we provide the seed programs generated by Codex in `TitanFuzz_codex_seed_programs.zip`. Note that we have filtered out the invalid programs via execution and in later experiments we will only use the valid ones (in the `fix` subfolders).


<details><summary>The structure of `codex_seed_programs`</summary>

```
- codex_seed_programs
    - codex_tf_seeds: contains the seed programs for tensorflow.
        - raw: contains the seed programs generated by codex.
            - tf.add: stores the 25 completions sampled from codex for each target API.
                - 1.py: each python file contains the prompt and raw completion.
                - ...
                - 25.py
            - ...
        - fix: contains the valid seed programs. We execute every raw outputs to filter out the invalid ones.
            - tf.add
                - 1.py
                - ...
                - 21.py
            - ...
    - codex_torch_seeds: contains the seed programs for PyTorch.
        - raw
        - fix
- titanfuzz
    - README.md: this file
    - ...
```
</details>

#### 2. Evolutionary Fuzzing

The main script for our evolutionary fuzzing algorithm is `ev_generation.py`.

<details><summary>Do an example run</summary>
We provide a **demo** script to run test generation, which will run fuzzing for two demo APIs, each with a one-minute fuzzing budget.

If you have already created docker container successfully, you can run the following commands to test on TensorFlow:
```
bash scripts/demo_run_tf.sh
```
If you see the following, you are running TitanFuzz successfully!
```
...
Copy finished
--------------Begin executing bash scripts/run.sh tf data/tf_apis_demo.txt------------
Current directory:  /home/src/run
Results will be dumped to:  /home/src/run/Results
[0 / 2] tf.nn.conv2d
...
```

The generated tests will be dumped to `titanfuzz:/home/src/run/Results/tf`.

For example, to view a list of the valid tensorflow programs generated by `TitanFuzz`, you may run:
```
docker exec -it titanfuzz ls src/run/Results/tf/valid/
```
To view a certain tensorflow program in the list (let's say `tf.nn.conv2d_7.py`), you may run:
```
# Remember to replace the file name with a real file name from the list above
docker exec -it titanfuzz cat src/run/Results/tf/valid/tf.nn.conv2d_7.py
```
Here is an example output program for `tf.nn.conv2d`:
```
input_data = np.random.rand(1, 3, 3, 1)
input_data = tf.constant(input_data, dtype=tf.float32)
filter_data = np.random.rand(2, 2, 1, 1)
filter_data = tf.constant(filter_data, dtype=tf.float32)
y = tf.nn.conv2d(input_data, filter_data, strides=[1, 1, 1, 1], padding='SAME')
```

To test on PyTorch, you may run:
```
bash scripts/demo_run_torch.sh
```
> Note that the running script will first copy the current folder to docker, and thus will remove the previous tensorflow generations under `Results/tf/`. We recommend testing one library (2. Evolutionary Fuzzing + 3. Bug detection) at a time,

<details><summary>Run without docker</summary>
If you don't have docker environment, and have successfully build our conda environment, run:

```
conda activate titanfuzz
bash scripts/demo_run_tf.sh false
bash scripts/demo_run_torch.sh false
```
If you see the following, you are running TitanFuzz successfully!
```
Warning: running in a non-docker environment!
Current directory:  /your/local/path/titanfuzz
Results will be dumped to:  /your/local/path/titanfuzz/Results
[0 / 2] tf.nn.conv2d
...
```
</details>

</details>

<details><summary>Run on 100 APIs</summary>

To run the test generation on the 100 sampled APIs used in our ablation study (**RQ2**), you may run:

```
bash scripts/ablation_run_tf.sh
bash scripts/ablation_run_torch.sh
```
> This could take 1~2 hours to finish.

Similarly, if you don't have docker support, you may run locally:

```
conda activate titanfuzz
bash scripts/ablation_run_tf.sh false
bash scripts/ablation_run_torch.sh false
```
</details>

<details><summary>Run full experiments</summary>

To run **full-scale** test generation (**RQ1** and **RQ3**), you may run:
```
bash scripts/dockerrun.sh bash scripts/run.sh tf
bash scripts/dockerrun.sh bash scripts/run.sh torch
```
> This could take >48 hours to finish on one GPU, as there are thousands of DL APIs in each DL library.

</details>


#### 3. Bug detection

<details><summary> Run oracle </summary>
To run our oracle for bug detection, you may run:

```
bash scripts/dockerrun.sh python driver.py --mode race --input Results/tf --tf --output Results/tf/trace.txt
bash scripts/dockerrun.sh python driver.py --mode race --input Results/torch --output Results/torch/trace.txt
```

The outputs and the candidate bugs will be logged in `titanfuzz:/home/src/run/Results/{tf, torch}/trace.txt`.

> Note that in order to run bug detection for a certain library, you may want to first run one of demo / ablation / full setting in the above section (2. Evolutionary Fuzzing) to generate test programs for the particular library.

</details>

## Reproduce figures in paper

Due to the high cost of running the full experiments, we have also provided a simple script based on intermediate data to produce the main figures (Figure 7 and 8) in the paper.

You may choose to create a python virtual environment to run the script:
```bash
pip install brokenaxes
python scripts/draw_trend.py
```
You will obtain Figure 7 (`coverage_trend.png`) and Figure 8 (`codex_temperature.png`) in the paper under `figs/`.

## Evidence of bug finding

In RQ3, we claimed TitanFuzz has detected 65 bugs in total for PyTorch and TensorFlow, with 55 already confirmed by developers. Out of those 55 confirmed bugs, 44 are confirmed to be previously unknown, and 21 are already fixed by developers.

We provide a list of confirmed bug reports in `TitanFuzz-PyTorch-confirmed-issues.csv` and `TitanFuzz-TensorFlow-confirmed-issues.csv`.

## Extension

`TitanFuzz` framework is general, and can be extended to other models and libraries. For example, our default LLM for mutation is `facebook/incoder-1B` (https://huggingface.co/facebook/incoder-1B), you may change `run.sh` and pass `--model_name facebook/incoder-6B` to `ev_generation.py` to use a larger model. You may also use other LLMs to replace Codex for seed generation.


## Contact

Authors information:

| Name               | Email Address               |
| ------------------ | --------------------------- |
| Yinlin Deng        | yinlind2@illinois.edu       |
| Chunqiu Steven Xia | chunqiu2@illinois.edu       |
| Haoran Peng        | hurrypeng@mail.ustc.edu.cn  |
| Chenyuan Yang      | cy54@illinois.edu           |
| Lingming Zhang     | lingming@illinois.edu       |
