#!/usr/bin/env bash

IDX_FOLD=$1
PATH_DATA=$2
DATA_ID=$3
TRAIN_STEP=$4
LAMBDA_MARK=${5:-0.2}
BATCH_SIZE=${6:-32}
NUM_LAYER=${7:-3}
LR=${8:-1.0}
WARM_STEP=${9:-8000}
MSG=${10:-""}
METRIC=${11:-"accuracy"}
DROP=${12:-0.3}
SRC_LEN=${13:-60}
TGT_LEN=${14:-100}
RAND=${15:-100}
VALIDSTEP=${16:-1000}

# rm ${PATH_DATA}/*.pt
# rm ${PATH_DATA}/*.log
# rm ${PATH_DATA}/*.txt
# rm ${PATH_DATA}/*.sh
# rm -r ${PATH_DATA}/tensorboard/ 

# rm -r ${PATH_DATA}/decoding-self-attn-debug
# rm -r ${PATH_DATA}/align-attn
# rm -r ${PATH_DATA}/self-attn-debug


me="$(basename "$(test -L "$0" && readlink "$0" || echo "$0")")"
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cp $DIR/$me ${PATH_DATA}/

echo $MSG > ${PATH_DATA}/msg.log
CUR_DIR=$(pwd)
CODE_DIR="../libs/opennmt_py_phraseRnn/" # for lstm  using cell state + hidden state 
cd $CODE_DIR && echo $CODE_DIR > ${PATH_DATA}/code.log && git diff >> ${PATH_DATA}/code.log && git log --oneline >> ${PATH_DATA}/code.log
cd $CUR_DIR

python ${CODE_DIR}/preprocess.py \
-train_src ${PATH_DATA}/X_train_${IDX_FOLD}.tsv \
-train_tgt ${PATH_DATA}/Y_train_${IDX_FOLD}.tsv \
-valid_src ${PATH_DATA}/X_dev_${IDX_FOLD}.tsv \
-valid_tgt ${PATH_DATA}/Y_dev_${IDX_FOLD}.tsv \
-save_data ${PATH_DATA}/${DATA_ID}.data${IDX_FOLD} \
-src_words_min_frequency 3 \
-tgt_words_min_frequency 3 \
-overwrite \
-src_seq_length $SRC_LEN -tgt_seq_length $TGT_LEN \
#-dynamic_dict -share_vocab 

# CUR_DIR=$(pwd)
# CODE_DIR="../opennmt_py_phraseDSViewDr/"
me="$(basename "$(test -L "$0" && readlink "$0" || echo "$0")")"
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

python  ${CODE_DIR}/train.py \
-data ${PATH_DATA}/${DATA_ID}.data${IDX_FOLD} -feat_merge sum \
-save_model ${PATH_DATA}/${DATA_ID}-model${IDX_FOLD}  \
-layers ${NUM_LAYER} -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8 -seed ${RAND} \
-encoder_type transformer -decoder_type transformer  -position_encoding \
-train_steps ${TRAIN_STEP}  -max_generator_batches 2 -dropout ${DROP} \
-batch_size ${BATCH_SIZE} -batch_type tokens -normalization tokens  -accum_count 8 -report_every $VALIDSTEP \
-optim adam -adam_beta1 0.9  -adam_beta2 0.98 -decay_method noam -warmup_steps ${WARM_STEP} -learning_rate ${LR} \
-max_grad_norm 0 -param_init 0  -param_init_glorot \
-label_smoothing 0.1 -valid_steps $VALIDSTEP  -keep_checkpoint 1 -save_checkpoint_steps $VALIDSTEP \
-log_file ${PATH_DATA}/train_${IDX_FOLD}.log  \
-early_stopping_criteria ${METRIC} -early_stopping 10 \
-tensorboard -tensorboard_log_dir ${PATH_DATA}/tensorboard/paralellFlatQK \
-world_size 1  -gpu_ranks 0 \
-write_config ${PATH_DATA}/config.yml \
-share_decoder_embeddings \
-gram_sizes 0 0 2 2 3 3 4 4 \
# -train_from ${PATH_DATA}/${DATA_ID}-model${IDX_FOLD}_step_62000.pt 
#-copy_attn -reuse_copy

for type_d in "dev" "test"; do 

    python ${CODE_DIR}/translate.py \
    -model ${PATH_DATA}/${DATA_ID}-model${IDX_FOLD}_step_0.pt \
    -src ${PATH_DATA}/X_${type_d}_${IDX_FOLD}.tsv \
    -tgt ${PATH_DATA}/Y_${type_d}_${IDX_FOLD}.tsv \
    -output ${PATH_DATA}/Y_pred_${IDX_FOLD}.tsv \
    -replace_unk -share_vocab \
    -report_time -verbose --log_file_level 0 \
    -log_file ${PATH_DATA}/${type_d}_${IDX_FOLD}.log  \
    -gpu 0 \
    # -attn_debug -self_attn_debug \

    # evaluate acc
    python ../src/eval_metrics.py \
    --path ${PATH_DATA} \
    --pred Y_pred_${IDX_FOLD}.tsv \
    --target Y_${type_d}_${IDX_FOLD}.tsv \
    > ${PATH_DATA}/result_${type_d}_${IDX_FOLD}.log

    python ${CODE_DIR}/../../data-sem/src/evaluate.py --path ${PATH_DATA}/ \
    --src X_${type_d}_${IDX_FOLD}.tsv \
    --tgt Y_${type_d}_${IDX_FOLD}.tsv \
    --pred Y_pred_${IDX_FOLD}.tsv \
    > ${PATH_DATA}/result_logic_${type_d}_${IDX_FOLD}.log

    # print result
    cat ${PATH_DATA}/result_${type_d}_${IDX_FOLD}.log

    cat ${PATH_DATA}/result_logic_${type_d}_${IDX_FOLD}.log
done  



cd ${PATH_DATA}/../src/ && \
python lf_smatch.py \
-pred ${CUR_DIR}/${PATH_DATA}/Y_pred_${IDX_FOLD}.tsv \
-test ${CUR_DIR}/${PATH_DATA}/Y_test_${IDX_FOLD}.tsv  \
> ${CUR_DIR}/${PATH_DATA}/result_smatch_core_${IDX_FOLD}.log && \
cd ${CUR_DIR} && cat ${PATH_DATA}/result_smatch_core_${IDX_FOLD}.log

#bash run_stats.sh
