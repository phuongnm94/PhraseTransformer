RUN_PREDICT=${1:-true}

CUDA_VISIBLE_DEVICES=0
USER_DIR="src/"
PATH_DATA=`dirname "$0"`
echo $PATH_DATA

MAX_LEN_B=200

source $PATH_DATA/setting.sh 

python src/avg_last_checkpoint.py --inputs ${PATH_DATA} --num-epoch-checkpoints 5 --output ${PATH_DATA}/averaged.pt

for TYPE_F in "test"  "val"  # "train" 
do 
    TEST_FILE="${PATH_DATA}/${TYPE_F}.${TGT_LANG}.tmp"
    echo $USER_DIR

    if $RUN_PREDICT ; then
        PATH_DATA_BIN=$PATH_DATA/infer-${TYPE_F}
        mkdir $PATH_DATA_BIN 

        cp $PATH_DATA/data-bin/dict.$SRC_LANG.txt  $PATH_DATA_BIN # $PATH_DATA/data-bin/dict.$TGT_LANG.txt

        # Binarize the dataset
        fairseq-preprocess \
            --source-lang $SRC_LANG --target-lang $TGT_LANG \
            --testpref $PATH_DATA/$RAW_DATA_PATH/${TYPE_F}.bpe \
            --destdir $PATH_DATA_BIN --thresholdtgt 0 --thresholdsrc 0 \
            --workers 32  --srcdict $PATH_DATA_BIN/dict.$SRC_LANG.txt --joined-dictionary  # --only-source

        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
        fairseq-generate ${PATH_DATA_BIN} \
            --source-lang $SRC_LANG --target-lang $TGT_LANG \
            --task translation --user-dir $USER_DIR \
            --path ${PATH_DATA}/averaged.pt  --beam 5 --remove-bpe \
        > ${PATH_DATA}/out.${TYPE_F}.tmp

        cat ${PATH_DATA}/out.${TYPE_F}.tmp | grep '^H' | sed 's/^H\-//g' | sort -t ' ' -k1,1 -n | cut -f 3- > ${PATH_DATA}/generated_averaged.${TYPE_F}.result
        cat ${PATH_DATA}/out.${TYPE_F}.tmp | grep '^T' | sed 's/^T\-//g' | sort -t ' ' -k1,1 -n | cut -f 2- > ${TEST_FILE}
        

        rm -r ${PATH_DATA_BIN}  
    fi 
        
    mkdir ${PATH_DATA}/tmp;  mv ${PATH_DATA}/*.tmp  ${PATH_DATA}/tmp 

done
