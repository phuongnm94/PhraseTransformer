
PATH_DATA=`dirname "$0"` # `dirname "$(readlink -f "$0")"`
echo $PATH_DATA

CUDA_VISIBLE_DEVICES=0
ARCH="transformer_iwslt_de_en"
USER_DIR="src/PhraseTransformer"
PHRASE_DROPOUT=0.5
WARMUP=6000
NUM_PHRASE_LAYERS=6

source $PATH_DATA/setting.sh 
echo $TEMPLATE_TYPE

PATH_DATA=$PATH_DATA/
PATH_DATA_BIN=$PATH_DATA/data-bin

# save diff in code
echo '' > $PATH_DATA/code.diff
for FILE_DIFF in $USER_DIR/*.py 
do 
    FILE_NAME="$(basename -- $FILE_DIFF)"
    git diff --no-index "src/PhraseTransformer_org/$FILE_NAME" $FILE_DIFF >> "$PATH_DATA/code.diff"
done

# Binarize the dataset
fairseq-preprocess \
    --source-lang $SRC_LANG --target-lang $TGT_LANG \
    --trainpref $PATH_DATA/../train --validpref $PATH_DATA/../dev --testpref $PATH_DATA/../test \
    --destdir $PATH_DATA_BIN --thresholdtgt 0 --thresholdsrc 0 \
    --workers 32 --joined-dictionary
 
echo $USER_DIR && ls $USER_DIR

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES fairseq-train \
    $PATH_DATA_BIN \
    --source-lang $SRC_LANG --target-lang $TGT_LANG  \
    --validate-after-updates 5000 --no-progress-bar \
    --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --adam-eps 1e-9  \
    --lr 7e-4 --lr-scheduler inverse_sqrt --warmup-updates $WARMUP \
    --dropout 0.3 --attention-dropout 0.1 --activation-dropout 0.1 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 --max-epoch 100 --seed 1 --share-all-embeddings --save-dir $PATH_DATA \
    --log-interval 100 --tensorboard-logdir $PATH_DATA/tensorboard \
    --keep-last-epochs 5 \
    --task  translation --user-dir $USER_DIR --arch $ARCH  --ngram-sizes $NGRAM_SIZES  --phrase-dropout $PHRASE_DROPOUT --num-phrase-layers $NUM_PHRASE_LAYERS \
    --update-freq 4 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    |& tee  $PATH_DATA/train.log



# # Evaluate
bash $PATH_DATA/generate.sh