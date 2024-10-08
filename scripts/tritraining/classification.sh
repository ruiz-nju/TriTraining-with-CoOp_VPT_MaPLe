DATA=/mnt/hdd/zhurui/data
TRAINER=TriTraining

DATASET=$1
SHOTS=$2
SEED=$3

DIR=output/${TRAINER}/classification/${DATASET}/shots_${SHOTS}/seed_${SEED}
CUDA_VISIBLE_DEVICES=1 python tritraining_main.py \
    --root ${DATA} \
    --seed ${SEED} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    TRAINER.MODAL classification \
    DATASET.SUBSAMPLE_CLASSES all 