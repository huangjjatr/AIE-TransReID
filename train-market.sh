#!/bin/bash
CONFIG="--config_file configs/Market/vit_transreid_stride_384.yml"
LOG_PATH="logs/market_attribute/"
WEIGHT="TEST.WEIGHT ${LOG_PATH}transformer.pth"
EPOCHS="SOLVER.MAX_EPOCHS 120 SOLVER.CHECKPOINT_PERIOD 120"

for i in {1..30}
do
    sie_coef=$(echo "scale=2; $i/10" | bc)
    SIE_COE="MODEL.SIE_COE $sie_coef"
    python train.py ${CONFIG} ${SIE_COE} OUTPUT_DIR ${LOG_PATH} MODEL.SIE_XISHU '3.0' $EPOCHS
    tail -5 ${LOG_PATH}train_log.txt | sed -n "s/.*INFO: \(.*\)/\1/p" >> "${LOG_PATH}train_bg_384_0.log"
    if [ $i -eq 26 ]; then
        mv ${LOG_PATH}transformer_120.pth ${LOG_PATH}transformer_0_26.pth
    fi
done
