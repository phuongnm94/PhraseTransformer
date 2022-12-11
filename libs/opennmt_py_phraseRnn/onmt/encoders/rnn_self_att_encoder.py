"""Define RNN-based encoders."""
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from onmt.encoders.rnn_encoder import RNNEncoder
from onmt.modules import MultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.utils.misc import sequence_mask


class RNNSelfAttentionEncoder(RNNEncoder):
    """ A generic recurrent neural network encoder.

    Args:
       rnn_type (str):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :class:`torch.nn.Dropout`
       embeddings (onmt.modules.Embeddings): embedding module to use
    """

    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, embeddings=None,
                 use_bridge=False):
        super(RNNSelfAttentionEncoder, self).__init__(rnn_type, bidirectional, num_layers,
                                                      hidden_size, dropout=dropout, embeddings=embeddings,
                                                      use_bridge=use_bridge)

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.self_attn_dropout = 0.1
        self.self_attn = MultiHeadedAttention(
            head_count=8, model_dim=hidden_size, dropout=self.self_attn_dropout,
            max_relative_positions=0)
        self.feed_forward = PositionwiseFeedForward(hidden_size, 2048, self.self_attn_dropout)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(self.self_attn_dropout)

    def forward(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        emb = self.embeddings(src)
        # s_len, batch, emb_dim = emb.size()

        inputs = emb.transpose(0, 1)
        mask = ~sequence_mask(lengths).unsqueeze(1)
        input_norm = self.layer_norm(inputs)
        context, _self_attn = self.self_attn(input_norm, input_norm, input_norm,
                                    mask=mask, attn_type="self")
        out = self.dropout(context) + inputs
        out = self.feed_forward(out)
        emb = out.transpose(0, 1)

        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Tensor.
            lengths_list = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths_list)

        memory_bank, encoder_final = self.rnn(packed_emb)

        if lengths is not None and not self.no_pack_padded_seq:
            memory_bank = unpack(memory_bank)[0]

        if self.use_bridge:
            encoder_final = self._bridge(encoder_final)
        return encoder_final, memory_bank, lengths


