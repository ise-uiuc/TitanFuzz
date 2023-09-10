TEST_LIB="$1"
INPUT=${2:-'data/'${TEST_LIB}'_apis.txt'}
CUDA_DEVICE=0
SEEDSEL=fitness
OPSEL=ts
SEED_POOL_SIZE=10
BATCH_SIZE=30
MAX_VALID=-1
TIMEOUT=60
SEEDFOLDER=/home/parent/codex_seed_programs/codex_${TEST_LIB}_seeds/fix
FOLDER=Results/${TEST_LIB}/
API=all
random_seed=420
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python ev_generation.py --library ${TEST_LIB} \
                        --seedfolder ${SEEDFOLDER} \
                        --folder ${FOLDER} \
                        --max_valid ${MAX_VALID} \
                        --timeout ${TIMEOUT} \
                        --batch_size ${BATCH_SIZE} \
                        --random_seed ${random_seed} \
                        --only_valid \
                        --seed_selection_algo ${SEEDSEL} \
                        --mutator_selection_algo ${OPSEL} \
                        --apilist ${INPUT} \
                        --api all \
                        --seed_pool_size ${SEED_POOL_SIZE} \
                        --relaxargmut
