VOCAB_FILE="data_bpe/vocab_zh.txt"

subword-nmt learn-bpe -s 16000  < raw/train.zh > $VOCAB_FILE
subword-nmt apply-bpe -c $VOCAB_FILE < raw/test.zh > data_bpe/test.zh
subword-nmt apply-bpe -c $VOCAB_FILE < raw/train.zh > data_bpe/train.zh
subword-nmt apply-bpe -c $VOCAB_FILE < raw/dev.zh > data_bpe/dev.zh

VOCAB_FILE="data_bpe/vocab_vi.txt"

subword-nmt learn-bpe -s 4000  < raw/train.vi > $VOCAB_FILE
subword-nmt apply-bpe -c $VOCAB_FILE < raw/test.vi > data_bpe/test.vi
subword-nmt apply-bpe -c $VOCAB_FILE < raw/train.vi > data_bpe/train.vi
subword-nmt apply-bpe -c $VOCAB_FILE < raw/dev.vi > data_bpe/dev.vi