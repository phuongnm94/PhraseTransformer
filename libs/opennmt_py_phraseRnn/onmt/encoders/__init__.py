"""Module defining encoders."""
from onmt.encoders.combined_transformer_rnn import CombinedTransformerRnnEncoder
from onmt.encoders.encoder import EncoderBase
from onmt.encoders.rnn_self_att_encoder import RNNSelfAttentionEncoder
from onmt.encoders.transformer import TransformerEncoder
from onmt.encoders.ggnn_encoder import GGNNEncoder
from onmt.encoders.rnn_encoder import RNNEncoder
from onmt.encoders.cnn_encoder import CNNEncoder
from onmt.encoders.mean_encoder import MeanEncoder
from onmt.encoders.audio_encoder import AudioEncoder
from onmt.encoders.image_encoder import ImageEncoder


str2enc = {"ggnn": GGNNEncoder, "rnn": RNNEncoder, "brnn": RNNEncoder,
           "cnn": CNNEncoder, "transformer": TransformerEncoder,  "rnn-selfattn": RNNSelfAttentionEncoder, 
           "transformer-rnn": CombinedTransformerRnnEncoder,
           "img": ImageEncoder, "audio": AudioEncoder, "mean": MeanEncoder}

__all__ = ["EncoderBase", "TransformerEncoder", "RNNEncoder", "CNNEncoder",
           "MeanEncoder", "str2enc"]
