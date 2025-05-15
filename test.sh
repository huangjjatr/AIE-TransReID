#!/bin/bash
PARAMS="MODEL.NUM_RULES 0 MODEL.SIE_XISHU 3.0 TEST.IMS_PER_BATCH 128  MODEL.PRETRAIN_CHOICE ''"

test_market(){
    CONFIG="--config_file configs/Market/vit_transreid_stride_384.yml"
    LOG_PATH="logs/market_attribute"
    MODEL_PATH="TEST.WEIGHT ${LOG_PATH}transformer_0_26.pth"
    ATTRIBUTES="mr$1"
    
    for i in {1..10}
    do
        cd text && echo "RUN: $i-th random drop out attibute text with ratios of $1 ..."
        python add_noise.py market_text.json $1 $ATTRIBUTES
        cd ..
        python test.py ${CONFIG}  MODEL.SIE_COE 2.6 $MODEL_PATH OUTPUT_DIR ${LOG_PATH} ${PARAMS} MODEL.TEXT $ATTRIBUTES
        tail -4 ${LOG_PATH}test_log.txt | sed -n "s/.*INFO: \(.*\)/\1/p" >> "${LOG_PATH}$ATTRIBUTES.log"
    done
}

test_duke(){
    CONFIG="--config_file configs/DukeMTMC/vit_transreid_stride_384.yml"
    LOG_PATH="logs/duke_attribute/"
    MODEL_PATH="TEST.WEIGHT ${LOG_PATH}transformer_0_16.pth"
    ATTRIBUTES="dr$1"

    for i in {1..10}
    do
        cd text && echo "RUN: $i-th random drop out attibute text with ratios of $1 ..."
        python add_noise.py duke_text.json $1 $ATTRIBUTES
        cd ..
        python test.py ${CONFIG} MODEL.SIE_COE 1.6 $MODEL_PATH OUTPUT_DIR ${LOG_PATH} ${PARAMS} MODEL.TEXT $ATTRIBUTES
        tail -4 ${LOG_PATH}test_log.txt | sed -n "s/.*INFO: \(.*\)/\1/p" >> "${LOG_PATH}$ATTRIBUTES.log"
    done
}

for sigma in "0.01" "0.02" "0.05" "0.1" "0.2"
do
    case $1 in
        m)
           test_market $sigma
        ;;
        d)
           test_duke $sigma
        ;;
        *)
       echo "无效选择参数，请选择'm' 或 'd'"
    esac
done