data_folder=data/JTB_dataset_0707/
VOCAB_FILE="$data_folder/music_vocab.txt"

subword-nmt learn-bpe -s 100  < $data_folder/train.music_in > $VOCAB_FILE
subword-nmt apply-bpe -c $VOCAB_FILE < $data_folder/test.music_in > $data_folder/test.bpe.music_in
subword-nmt apply-bpe -c $VOCAB_FILE < $data_folder/train.music_in > $data_folder/train.bpe.music_in
subword-nmt apply-bpe -c $VOCAB_FILE < $data_folder/val.music_in > $data_folder/val.bpe.music_in


# subword-nmt learn-bpe -s 4000  < data_music/JTB_dataset_0707/train.music_out > $VOCAB_FILE
subword-nmt apply-bpe -c $VOCAB_FILE < $data_folder/test.music_out > $data_folder/test.bpe.music_out
subword-nmt apply-bpe -c $VOCAB_FILE < $data_folder/train.music_out > $data_folder/train.bpe.music_out
subword-nmt apply-bpe -c $VOCAB_FILE < $data_folder/val.music_out > $data_folder/val.bpe.music_out