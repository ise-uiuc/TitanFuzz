#!/bin/bash
# Build and initialize container

my_image=titanfuzz:latest
my_container=titanfuzz
docker run --gpus all -itd --name $my_container $my_image

# Grant permission for downloading transformer models and others
docker exec -it $my_container mkdir /.cache
docker exec -it $my_container chmod 777 /.cache
docker exec -it $my_container mkdir /.local
docker exec -it $my_container chmod 777 /.local

# Copy seed programs to /home/parent
codex_seed_programs_path=../codex_seed_programs/
docker exec -it -w /home/ $my_container mkdir parent
docker cp $codex_seed_programs_path $my_container:/home/parent/
docker exec -it -w /home/parent $my_container ls
