#!/bin/bash
my_container=titanfuzz
dest_dir=/home/src/run/
echo $dest_dir
docker exec -it -w /home/ $my_container rm -r $dest_dir
docker exec -it -w /home/ $my_container mkdir -p $dest_dir
docker cp ./ $my_container:$dest_dir

# Change the permission of source code to r-x
docker exec -it -w $dest_dir $my_container chmod 777 $(find . -type d)
docker exec -it -w $dest_dir $my_container chmod 555 $(find . -type f)

docker exec -it --user root -w $dest_dir $my_container mkdir -p Results
docker exec -it --user root -w $dest_dir $my_container chmod -R 777 Results

echo "Copy finished"
