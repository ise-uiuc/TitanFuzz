rm -r Results/torch
DOCKER_RUN=${1:-true}
if $DOCKER_RUN ; then
    bash scripts/copyfile.sh
    bash scripts/dockerrun.sh bash scripts/run.sh torch data/torch_valid_apis_100sample.txt
else
    echo "Warning: running in a non-docker environment!"
    bash scripts/local_run.sh torch data/torch_valid_apis_100sample.txt
fi
