
from fairseq.models.transformer import TransformerEncoder, TransformerModel

import torch.nn.functional as F

from fairseq.modules import (
    TransformerEncoderLayer, 
)

from fairseq.models import (
    register_model,
    register_model_architecture,
)


from fairseq.models.transformer import (
    base_architecture, transformer_iwslt_de_en, transformer_wmt_en_de,
    transformer_vaswani_wmt_en_de_big, transformer_vaswani_wmt_en_fr_big, transformer_wmt_en_de_big, transformer_wmt_en_de_big_t2t
)

from .phrase_multihead_attention import PhraseMultiheadAttention


@register_model('phrase_transformer')
class PhraseTransformerModel(TransformerModel): 
 
    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        parser.add_argument('--phrase-dropout', type=float,
                            default=0.5,
                            help='template dropout rate')    
        parser.add_argument('--ngram-sizes',
                            type=int, nargs='+', default=None,
                            help='define gram sizes for heads in multihead transformer')    
        parser.add_argument('--num-phrase-layers',   
                            type=int, default=None,
                            help='define num layers appling phrase mechanism')    
 
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return PhraseTransformerEncoder(args, src_dict, embed_tokens)


class PhraseTransformerEncoder(TransformerEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        args_setting = args[0]
        if args_setting.num_phrase_layers is not None and args_setting.num_phrase_layers < args_setting.encoder_layers:
            if args_setting.num_phrase_layers >= 0:
                for i in range(args_setting.num_phrase_layers, args_setting.encoder_layers):
                    self.layers[i] = TransformerEncoderLayer(args_setting) 
            else:
                for i in range(0, args_setting.encoder_layers + args_setting.num_phrase_layers):
                    self.layers[i] = TransformerEncoderLayer(args_setting) 
 
    def build_encoder_layer(self, args):
        return PhraseTransformerEncoderLayer(args)


class PhraseTransformerEncoderLayer(TransformerEncoderLayer):

    def build_self_attention(self, embed_dim, args):
        return PhraseMultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            ngram_sizes=args.ngram_sizes,
            phrase_dropout=args.phrase_dropout,
            num_phrase_layers=args.num_phrase_layers,
        )
 
@register_model_architecture('phrase_transformer', 'phrase_transformer')
def phrase_transformer_base_architecture(args):
    base_architecture(args)


@register_model_architecture('phrase_transformer', 'phrase_transformer_iwslt_de_en')
def phrase_transformer_iwslt_de_en(args):
    transformer_iwslt_de_en(args)


@register_model_architecture('phrase_transformer', 'phrase_transformer_wmt_en_de')
def phrase_transformer_wmt_en_de(args):
    transformer_wmt_en_de(args)


# parameters used in the "Attention Is All You Need" paper (Vaswani, et al, 2017)
@register_model_architecture('phrase_transformer', 'phrase_transformer_vaswani_wmt_en_de_big')
def phrase_transformer_vaswani_wmt_en_de_big(args):
    transformer_vaswani_wmt_en_de_big(args)


@register_model_architecture('phrase_transformer', 'phrase_transformer_vaswani_wmt_en_fr_big')
def phrase_transformer_vaswani_wmt_en_fr_big(args):
    transformer_vaswani_wmt_en_fr_big(args)


@register_model_architecture('phrase_transformer', 'phrase_transformer_wmt_en_de_big')
def phrase_transformer_wmt_en_de_big(args):
    transformer_wmt_en_de_big(args)


# default parameters used in tensor2tensor implementation
@register_model_architecture('phrase_transformer', 'phrase_transformer_wmt_en_de_big_t2t')
def phrase_transformer_wmt_en_de_big_t2t(args):
    transformer_wmt_en_de_big_t2t(args)
