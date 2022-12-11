#!/usr/bin/env bash

IDX_FOLD=$1
PATH_DATA=$2
DATA_ID=$3
TRAIN_STEP=$4
LAMBDA_MARK=${5:=0.2}
BATCH_SIZE=${6:=32}
N_BEST=${7:=5}
TYPE_EVAl=${8:-'test'}
 
CODE_FOLDER="../libs/opennmt_py_phraseRnn/"  
 

python ${CODE_FOLDER}/translate.py \
-model ${PATH_DATA}/${DATA_ID}-model${IDX_FOLD}_step_${TRAIN_STEP}.pt \
-src ${PATH_DATA}/X_${TYPE_EVAl}_${IDX_FOLD}.tsv \
-output ${PATH_DATA}/Y_pred_${TYPE_EVAl}_${IDX_FOLD}.tsv \
-tgt ${PATH_DATA}/Y_${TYPE_EVAl}_${IDX_FOLD}.tsv \
-replace_unk -share_vocab \
-report_time --log_file_level 0 \
-log_file ${PATH_DATA}/${TYPE_EVAl}_${IDX_FOLD}.log  \
-batch_size ${BATCH_SIZE} \
-n_best ${N_BEST} -verbose  \
-gpu 0 \
# -attn_debug  \
# -self_attn_debug \

# # #

CUR_DIR=$(pwd)

# evaluate acc
python  ${CUR_DIR}/../src/em_evaluate.py \
--path ${PATH_DATA} \
--pred Y_pred_${TYPE_EVAl}_${IDX_FOLD}.tsv \
--target Y_${TYPE_EVAl}_${IDX_FOLD}.tsv \
> ${PATH_DATA}/result_translate_${TYPE_EVAl}_${IDX_FOLD}.log

#me="$(basename "$(test -L "$0" && readlink "$0" || echo "$0")")"
#DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
#cp $DIR/$me ${PATH_DATA}/

# print result
cat ${PATH_DATA}/result_translate_${TYPE_EVAl}_${IDX_FOLD}.log

python ${CUR_DIR}/../src/logic_evaluate.py --path ${PATH_DATA}/ \
--src  X_${TYPE_EVAl}_${IDX_FOLD}.tsv \
--tgt  Y_${TYPE_EVAl}_${IDX_FOLD}.tsv \
--pred  Y_pred_${TYPE_EVAl}_${IDX_FOLD}.tsv \
--n_best  ${N_BEST} \
> ${PATH_DATA}/result_logic_${TYPE_EVAl}_${IDX_FOLD}_step_${TRAIN_STEP}.log 
cat ${PATH_DATA}/result_logic_${TYPE_EVAl}_${IDX_FOLD}_step_${TRAIN_STEP}.log
  

