# PhraseTransformer
## Abstract
  Semantic parsing is a challenging task mapping a natural language utterance to machine-understandable information representation. 
  Recently, approaches using neural machine translation (NMT) have achieved many promising results, especially the Transformer. %, because of the ability to learn long-range word dependencies.  
  However, the typical drawback of adapting the vanilla Transformer to semantic parsing is that it does not consider the phrase in expressing the information of sentences while phrases play an important role in constructing the sentence meaning. 
  Therefore, we propose an architecture, PhraseTransformer, that is capable of a more detailed meaning representation by learning the phrase dependencies in the sentence. 
  The main idea is to incorporate Long Short-Term Memory  into the Self-Attention mechanism of the original Transformer to capture the local context of a word.
 To this end, our proposed model improved  in understand 
  Experimental results show that our proposed model 
  performs better than the original Transformer  in terms of understanding sentences structure as well as logical representation and raises the model local context-awareness   without any support from external tree information. 
  Besides, although the recurrent architecture is integrated, the number of sequential operations of the PhraseTransformer is still $\mathcal{O}(1)$ similar to the original Transformer.  
  Our proposed model achieves strong competitive performance on Geo and MSParS datasets, and leads to SOTA performance on the Atis dataset for methods using neural networks. 
  In addition, to prove the generalization of our proposed model, we also conduct extensive experiments on three translation datasets IWLST14 German-English, IWSLT15 Vietnamese-English, WMT14 English-German, and show significant improvement. 
## Env
## Runs
## Citation
##  License
MIT-licensed. 