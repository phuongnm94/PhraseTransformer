DATA_DIR="../data-sem/atis/" && bash ./train_phrase_trans.sh  5 $DATA_DIR "atis" 100000 0.1 4096 6 0.1 100 "phrase fname " accuracy 0.1 60 100 100 1000 > $DATA_DIR/run.log
