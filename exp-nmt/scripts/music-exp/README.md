Music experiments steps:
1. Install environment with file `exp-nmt/requirements.txt`
2. Run file `data/parse.ipynb` to parse csv data to text data 
3. Run `vocab_generate.sh` to generate music notes vocab file
4. Create folder for each setting for example: `PhraseTransformerScripts/` or `TransformerScripts/` with modification of setting such as: `RAW_DATA_PATH`, `NGRAM_SIZES`, etc.
5. Run `PhraseTransformerScripts/run_sota_phrasetrans.sh` to train a PhraseTransformer model and generate a prediction of test and valid files (`generated_averaged.val.result`, `generated_averaged.test.result`) in corresponding setting folder.
