# PhraseTransformer
## Abstract
  Semantic parsing is a challenging task mapping a natural language utterance to machine-understandable information representation. 
  Recently, approaches using neural machine translation (NMT) have achieved many promising results, especially the Transformer.
  However, the typical drawback of adapting the vanilla Transformer to semantic parsing is that it does not consider the phrase in expressing the information of sentences while phrases play an important role in constructing the sentence meaning. 
  Therefore, we propose an architecture, PhraseTransformer, that is capable of a more detailed meaning representation by learning the phrase dependencies in the sentence. 
  The main idea is to incorporate Long Short-Term Memory  into the Self-Attention mechanism of the original Transformer to capture the local context of a word. Experimental results show that our proposed model  performs better than the original Transformer  in terms of understanding sentences structure as well as logical representation and raises the model local context-awareness   without any support from external tree information. 
  Besides, although the recurrent architecture is integrated, the number of sequential operations of the PhraseTransformer is still $\mathcal{O}(1)$ similar to the original Transformer.   Our proposed model achieves strong competitive performance on Geo and MSParS datasets, and leads to SOTA performance on the Atis dataset for methods using neural networks. 
  In addition, to prove the generalization of our proposed model, we also conduct extensive experiments on three translation datasets IWLST14 German-English, IWSLT15 Vietnamese-English, WMT14 English-German, and show significant improvement. 
## Env
**Prepare data and environments** build virtual env python with name: `env_phrase_trans`  and install all package in the `requirements.txt`:
```commandline
conda  create --prefix env_phrase_trans python=3.8 
source activate env_phrase_trans/

pip install torch==1.11.0+cu115 torchaudio==0.11.0+cu115 torchvision==0.12.0+cu115 --extra-index-url https://download.pytorch.org/whl/cu115

pip install -r requirements.txt 
```

## Runs
check all runs in `./exp-run/`
```
cd exp-run/
bash exp_phrase_trans.sh
```
## Citation
```bib
@Article{Nguyen2022,
author={Nguyen, Phuong Minh
and Le, Tung
and Nguyen, Huy Tien
and Tran, Vu
and Nguyen, Minh Le},
title={PhraseTransformer: an incorporation of local context information into sequence-to-sequence semantic parsing},
journal={Applied Intelligence},
year={2022},
month={Nov},
day={29},
issn={1573-7497},
doi={10.1007/s10489-022-04246-0},
url={https://doi.org/10.1007/s10489-022-04246-0}
}
```
##  License
MIT-licensed. 
