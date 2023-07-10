RUN_PREDICT=${1:-true}

PATH_DATA=`dirname "$0"` # `dirname "$(readlink -f "$0")"`
echo $PATH_DATA
USER_DIR="src/"

source $PATH_DATA/setting.sh 
echo $TEMPLATE_TYPE


MOSES_LIB="../nmt/mosesdecoder"
TEST_FILE=$PATH_DATA/test.debpe.$TGT_LANG


# run avg last 5 checkpoint 
if $RUN_PREDICT ; then
    python src/avg_last_checkpoint.py --inputs ${PATH_DATA} --num-epoch-checkpoints 5 --output ${PATH_DATA}/averaged.pt
    fairseq-generate ${PATH_DATA}/data-bin \
            --source-lang $SRC_LANG --target-lang $TGT_LANG \
            --task translation --user-dir $USER_DIR \
            --path ${PATH_DATA}/averaged.pt  --beam 5  > ${PATH_DATA}/out.avg.log 
    cat ${PATH_DATA}/out.avg.log  | grep '^H' | sed 's/^H\-//g' | sort -t ' ' -k1,1 -n | cut -f 3- > ${PATH_DATA}/generated.result
    cat ${PATH_DATA}/out.avg.log  | grep '^T' | sed 's/^T\-//g' | sort -t ' ' -k1,1 -n | cut -f 2- > ${TEST_FILE} 
fi
cat ${PATH_DATA}/generated.result | ${MOSES_LIB}/scripts/generic/multi-bleu.perl ${TEST_FILE} > ${PATH_DATA}/log_avg_multi-bleu.log

# # run best checkpoint 
# if $RUN_PREDICT ; then
#     fairseq-generate ${PATH_DATA}/data-bin \
#             --source-lang $SRC_LANG --target-lang $TGT_LANG \
#             --task translation  --user-dir $USER_DIR \
#             --path ${PATH_DATA}/checkpoint_best.pt  --beam 5 --remove-bpe >  ${PATH_DATA}/out.best.log 
#     cat  ${PATH_DATA}/out.best.log | grep '^H' | sed 's/^H\-//g' | sort -t ' ' -k1,1 -n | cut -f 3- > ${PATH_DATA}/generated_best.result
# fi
# cat ${PATH_DATA}/generated_best.result | ${MOSES_LIB}/scripts/generic/multi-bleu.perl ${TEST_FILE} >${PATH_DATA}/log1best_multi-bleu.log
