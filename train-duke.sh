#!/bin/bash
CONFIG="--config_file configs/DukeMTMC/vit_transreid_stride_384.yml"
LOG_PATH="logs/duke_attribute/"
WEIGHT="TEST.WEIGHT ${LOG_PATH}transformer.pth"
EPOCHS="SOLVER.MAX_EPOCHS 120 SOLVER.CHECKPOINT_PERIOD 120"

for i in {1..15}
do
    sie_coef=$(echo "$i" | awk '{printf"%.2f", $1/5}')
    SIE_COE="MODEL.SIE_COE $sie_coef TEST.IMS_PER_BATCH 128"
    python train.py ${CONFIG} ${SIE_COE} OUTPUT_DIR ${LOG_PATH} MODEL.SIE_XISHU '3.0' $EPOCHS
    tail -6 ${LOG_PATH}train_log.txt >> "${LOG_PATH}train_384x128_0.log"
    if [ $i -eq 8 ]; then
        mv ${LOG_PATH}transformer_120.pth ${LOG_PATH}transformer_0_16.pth
    fi
done