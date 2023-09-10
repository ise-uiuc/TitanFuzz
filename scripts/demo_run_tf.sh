DOCKER_RUN=${1:-true}
if $DOCKER_RUN ; then
    bash scripts/copyfile.sh
    bash scripts/dockerrun.sh bash scripts/run.sh tf data/tf_apis_demo.txt
else
    echo "Warning: running in a non-docker environment!"
    bash scripts/local_run.sh tf data/tf_apis_demo.txt
fi
