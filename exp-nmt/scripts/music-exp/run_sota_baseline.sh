
PATH_DATA=`dirname "$0"` # `dirname "$(readlink -f "$0")"`
RAW_DATA_PATH="."
echo $PATH_DATA 

CUDA_VISIBLE_DEVICES=0
ARCH="transformer_iwslt_de_en"
SEED=1


source $PATH_DATA/setting.sh 
echo $TEMPLATE_TYPE
echo $RAW_DATA_PATH 

PATH_DATA=$PATH_DATA/
PATH_DATA_BIN=$PATH_DATA/data-bin
USER_DIR="src/fairseqSyntaxNMT"


# Binarize the dataset
fairseq-preprocess \
    --source-lang $SRC_LANG --target-lang $TGT_LANG \
    --trainpref $PATH_DATA/$RAW_DATA_PATH/train.bpe --validpref $PATH_DATA/$RAW_DATA_PATH/val.bpe --testpref $PATH_DATA/$RAW_DATA_PATH/test.bpe \
    --destdir $PATH_DATA_BIN --thresholdtgt 0 --thresholdsrc 0 \
    --workers 32 --joined-dictionary
 

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES fairseq-train \
    $PATH_DATA_BIN \
    --source-lang $SRC_LANG --target-lang $TGT_LANG  \
    --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --adam-eps 1e-9  \
    --lr 7e-5 --lr-scheduler inverse_sqrt --warmup-updates 200 \
    --dropout 0.3 --attention-dropout 0.1 --activation-dropout 0.1 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096   --max-epoch 100 --seed $SEED --share-all-embeddings --save-dir $PATH_DATA \
    --log-interval 100   --tensorboard-logdir $PATH_DATA/tensorboard \
    --keep-last-epochs 5 --keep-best-checkpoints 0 \
    --task  translation --arch $ARCH \
    --update-freq 4 \
    --best-checkpoint-metric loss 


# # Evaluate
bash $PATH_DATA/generate.sh
bash $PATH_DATA/eval_translation_baseline.sh