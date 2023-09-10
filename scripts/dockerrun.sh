#!/bin/bash
my_container=titanfuzz
dest_dir=/home/src/run/
cuda_visible_device=0
echo "--------------Begin executing "$@"---------------"
docker exec -it -w $dest_dir \
    -e CUDA_VISIBLE_DEVICES=${cuda_visible_device} \
    -e TF_FORCE_GPU_ALLOW_GROWTH=true \
    -e TF_CPP_MIN_LOG_LEVEL='3' \
    -e LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:/usr/local/cuda-11.0/targets/x86_64-linux/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    --user 1000:1000 \
    $my_container "$@"

echo "--------------Finish executing ---------------"

docker exec -it -w $dest_dir $my_container ls
