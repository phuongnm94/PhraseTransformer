"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn

from onmt.decoders.decoder import DecoderBase, RNNDecoderBase
from onmt.encoders.combined_transformer_rnn import EmbeddingSkipped
from onmt.models.stacked_rnn import StackedLSTM, StackedGRU
from onmt.modules import MultiHeadedAttention, AverageAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.utils.misc import sequence_mask


class CombinedTransformerRnnDecoderLayer(nn.Module):
    """
    Args:
      d_model (int): the dimension of keys/values/queries in
          :class:`MultiHeadedAttention`, also the input size of
          the first-layer of the :class:`PositionwiseFeedForward`.
      heads (int): the number of heads for MultiHeadedAttention.
      d_ff (int): the second-layer of the :class:`PositionwiseFeedForward`.
      dropout (float): dropout probability.
      self_attn_type (string): type of self-attention scaled-dot, average
    """

    def __init__(self, d_model, heads, d_ff, dropout, attention_dropout,
                 self_attn_type="scaled-dot", max_relative_positions=0,
                 aan_useffn=False, full_context_alignment=False,
                 alignment_heads=None):
        super(CombinedTransformerRnnDecoderLayer, self).__init__()

        if self_attn_type == "scaled-dot":
            self.self_attn = MultiHeadedAttention(
                heads, d_model, dropout=dropout,
                max_relative_positions=max_relative_positions)
        elif self_attn_type == "average":
            self.self_attn = AverageAttention(d_model,
                                              dropout=attention_dropout,
                                              aan_useffn=aan_useffn)

        self.context_attn = MultiHeadedAttention(
            heads, d_model, dropout=attention_dropout)

        self.state = {}
        self.feed_rnn_decoder = InputFeedRNNDecoder2("LSTM", bidirectional_encoder=False, num_layers=1,
                                                     hidden_size=d_model, attn_type="none",
                                                     copy_attn=False, dropout=0.01, embeddings=EmbeddingSkipped(d_model),
                                                     reuse_copy_attn=False, copy_attn_type="none")

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)
        self.full_context_alignment = full_context_alignment
        self.alignment_heads = alignment_heads
        self.self_attn_data = None
        self.save_self_attn = False

    def clean_self_attn_data(self):
        self.self_attn_data = None

    def forward(self, *args, **kwargs):
        """ Extend _forward for (possibly) multiple decoder pass:
        1. Always a default (future masked) decoder forward pass,
        2. Possibly a second future aware decoder pass for joint learn
            full context alignement.

        Args:
            * All arguments of _forward.
            with_align (bool): whether return alignment attention.

        Returns:
            (FloatTensor, FloatTensor, FloatTensor or None):

            * output ``(batch_size, 1, model_dim)``
            * top_attn ``(batch_size, 1, src_len)``
            * attn_align ``(batch_size, 1, src_len)`` or None
        """
        with_align = kwargs.pop('with_align', False)
        output, attns = self._forward(*args, **kwargs)
        top_attn = attns[:, 0, :, :].contiguous()
        attn_align = None
        if with_align:
            if self.full_context_alignment:
                # return _, (B, Q_len, K_len)
                _, attns = self._forward(*args, **kwargs, future=True)

            if self.alignment_heads is not None:
                attns = attns[:, :self.alignment_heads, :, :].contiguous()
            # layer average attention across heads, get ``(B, Q, K)``
            # Case 1: no full_context, no align heads -> layer avg baseline
            # Case 2: no full_context, 1 align heads -> guided align
            # Case 3: full_context, 1 align heads -> full cte guided align
            attn_align = attns.mean(dim=1)
        return output, top_attn, attn_align

    def _forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask,
                 layer_cache=None, step=None, future=False):
        """ A naive forward pass for transformer decoder.
        # TODO: change 1 to T as T could be 1 or tgt_len
        Args:
            inputs (FloatTensor): ``(batch_size, 1, model_dim)``
            memory_bank (FloatTensor): ``(batch_size, src_len, model_dim)``
            src_pad_mask (LongTensor): ``(batch_size, 1, src_len)``
            tgt_pad_mask (LongTensor): ``(batch_size, 1, 1)``

        Returns:
            (FloatTensor, FloatTensor):

            * output ``(batch_size, 1, model_dim)``
            * attns ``(batch_size, head, 1, src_len)``

        """
        dec_mask = None

        if step is None:
            tgt_len = tgt_pad_mask.size(-1)
            if not future:  # apply future_mask, result mask in (B, T, T)
                future_mask = torch.ones(
                    [tgt_len, tgt_len],
                    device=tgt_pad_mask.device,
                    dtype=torch.uint8)
                future_mask = future_mask.triu_(1).view(1, tgt_len, tgt_len)
                # BoolTensor was introduced in pytorch 1.2
                try:
                    future_mask = future_mask.bool()
                except AttributeError:
                    pass
                dec_mask = torch.gt(tgt_pad_mask + future_mask, 0)
            else:  # only mask padding, result mask in (B, 1, T)
                dec_mask = tgt_pad_mask

        input_norm = self.layer_norm_1(inputs)

        if isinstance(self.self_attn, MultiHeadedAttention):
            query, self_attn_data = self.self_attn(input_norm, input_norm, input_norm,
                                      mask=dec_mask,
                                      layer_cache=layer_cache,
                                      attn_type="self")
        elif isinstance(self.self_attn, AverageAttention):
            query, self_attn_data = self.self_attn(input_norm, mask=dec_mask,
                                      layer_cache=layer_cache, step=step)

        # save self attention data
        if self.save_self_attn and step is not None:
            self.self_attn_data = self_attn_data

        query = self.drop(query) + inputs

        query_norm = self.layer_norm_2(query)
        mid, attns = self.context_attn(memory_bank, memory_bank, query_norm,
                                       mask=src_pad_mask,
                                       layer_cache=layer_cache,
                                       attn_type="context")
        mid = self.feed_rnn_decoder(mid, memory_bank.transpose(0, 1), step=step)
        mid = mid.transpose(0, 1)

        output = self.feed_forward(self.drop(mid) + query)

        return output, attns

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.context_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.drop.p = dropout


class InputFeedRNNDecoder2(RNNDecoderBase):
    """Input feeding based decoder.

    See :class:`~onmt.decoders.decoder.RNNDecoderBase` for options.

    Based around the input feeding approach from
    "Effective Approaches to Attention-based Neural Machine Translation"
    :cite:`Luong2015`


    .. mermaid::

       graph BT
          A[Input n-1]
          AB[Input n]
          subgraph RNN
            E[Pos n-1]
            F[Pos n]
            E --> F
          end
          G[Encoder]
          H[memory_bank n-1]
          A --> E
          AB --> F
          E --> H
          G --> H
    """

    def forward(self, transform_emb, memory_bank, memory_lengths=None, step=None,
                **kwargs):
        """
        Args:
            tgt (LongTensor): sequences of padded tokens
                 ``(tgt_len, batch, nfeats)``.
            memory_bank (FloatTensor): vectors from the encoder
                 ``(src_len, batch, hidden)``.
            memory_lengths (LongTensor): the padded source lengths
                ``(batch,)``.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * dec_outs: output from the decoder (after attn)
              ``(tgt_len, batch, hidden)``.
        """

        dec_state, dec_outs  = self._run_forward_pass(transform_emb, memory_bank, memory_lengths)

        # Update the state with the result.
        if not isinstance(dec_state, tuple):
            dec_state = (dec_state,)
        self.state["hidden"] = dec_state
        self.state["input_feed"] = dec_outs[-1].unsqueeze(0)

        # Concatenates sequence of tensors along a new dimension.
        # NOTE: v0.3 to 0.4: dec_outs / attns[*] may not be list
        #       (in particular in case of SRU) it was not raising error in 0.3
        #       since stack(Variable) was allowed.
        #       In 0.4, SRU returns a tensor that shouldn't be stacke
        if type(dec_outs) == list:
            dec_outs = torch.stack(dec_outs)

        return dec_outs

    def _run_forward_pass(self, transform_emb, memory_bank, memory_lengths=None):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        """
        # Additional args check.
        batch_size = memory_bank.size(1)
        self.state["coverage"] = None
        self.state["input_feed"] = \
            memory_bank.transpose(0,1)[0].data.new(batch_size, self.hidden_size)\
                .zero_().unsqueeze(0)
        self.state["hidden"] = (
            memory_bank.transpose(0,1)[0].data.new(batch_size, self.hidden_size) \
                .zero_().unsqueeze(0),
            memory_bank.transpose(0,1)[0].data.new(batch_size, self.hidden_size) \
                .zero_().unsqueeze(0))
        input_feed = self.state["input_feed"].squeeze(0)

        input_feed_batch, _ = input_feed.size()
        # aeq(tgt_batch, input_feed_batch)
        # END Additional args check.

        dec_outs = []

        emb = transform_emb.transpose(0,1)
        assert emb.dim() == 3  # len x batch x embedding_dim

        dec_state = self.state["hidden"]

        # Input feed concatenates hidden state with
        # input at every time step.
        for emb_t in emb.split(1):
            decoder_input = torch.cat([emb_t.squeeze(0), input_feed], 1)
            rnn_output, dec_state = self.rnn(decoder_input, dec_state)

            decoder_output = rnn_output
            # decoder_output = self.dropout(decoder_output)
            input_feed = decoder_output
            dec_outs += [decoder_output]

        return dec_state, dec_outs

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        assert rnn_type != "SRU", "SRU doesn't support input feed! " \
                                  "Please set -input_feed 0!"
        stacked_cell = StackedLSTM if rnn_type == "LSTM" else StackedGRU
        return stacked_cell(num_layers, input_size, hidden_size, dropout)

    @property
    def _input_size(self):
        """Using input feed by concatenating input with attention vectors."""
        return self.embeddings.embedding_size + self.hidden_size

    def update_dropout(self, dropout):
        self.dropout.p = dropout
        self.rnn.dropout.p = dropout
        self.embeddings.update_dropout(dropout)


class CombinedTransformerRnnDecoder(DecoderBase):
    """The Transformer decoder from "Attention is All You Need".
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          BB[multi-head src-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> BB
          BB --> C
          C --> O


    Args:
       num_layers (int): number of encoder layers.
       d_model (int): size of the model
       heads (int): number of heads
       d_ff (int): size of the inner FF layer
       copy_attn (bool): if using a separate copy attention
       self_attn_type (str): type of self-attention scaled-dot, average
       dropout (float): dropout parameters
       embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings
    """

    def __init__(self, num_layers, d_model, heads, d_ff,
                 copy_attn, self_attn_type, dropout, attention_dropout,
                 embeddings, max_relative_positions, aan_useffn,
                 full_context_alignment, alignment_layer,
                 alignment_heads=None):
        super(CombinedTransformerRnnDecoder, self).__init__()

        self.embeddings = embeddings

        # Decoder State
        self.state = {}

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.transformer_layers = nn.ModuleList(
            [CombinedTransformerRnnDecoderLayer(d_model, heads, d_ff, dropout,
                                     attention_dropout, self_attn_type=self_attn_type,
                                     max_relative_positions=max_relative_positions,
                                     aan_useffn=aan_useffn,
                                     full_context_alignment=full_context_alignment,
                                     alignment_heads=alignment_heads)
             for i in range(num_layers)])

        # previously, there was a GlobalAttention module here for copy
        # attention. But it was never actually used -- the "copy" attention
        # just reuses the context attention.
        self._copy = copy_attn
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.alignment_layer = alignment_layer
        self.hidden_size = d_model

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.dec_layers,
            opt.dec_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.copy_attn,
            opt.self_attn_type,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0] if type(opt.attention_dropout)
                                        is list else opt.dropout,
            embeddings,
            opt.max_relative_positions,
            opt.aan_useffn,
            opt.full_context_alignment,
            opt.alignment_layer,
            alignment_heads=opt.alignment_heads)

    def init_state(self, src, memory_bank, enc_hidden):
        """Initialize decoder state."""
        self.state["src"] = src
        self.state["cache"] = None

    def map_state(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        self.state["src"] = fn(self.state["src"], 1)
        if self.state["cache"] is not None:
            _recursive_map(self.state["cache"])

    def detach_state(self):
        self.state["src"] = self.state["src"].detach()

    def forward(self, tgt, memory_bank, step=None, **kwargs):
        """Decode, possibly stepwise."""
        if step == 0:
            self._init_cache(memory_bank)

        tgt_words = tgt[:, :, 0].transpose(0, 1)

        emb = self.embeddings(tgt, step=step)
        assert emb.dim() == 3  # len x batch x embedding_dim

        output = emb.transpose(0, 1).contiguous()
        src_memory_bank = memory_bank.transpose(0, 1).contiguous()

        pad_idx = self.embeddings.word_padding_idx
        src_lens = kwargs["memory_lengths"]
        src_max_len = self.state["src"].shape[0]
        src_pad_mask = ~sequence_mask(src_lens, src_max_len).unsqueeze(1)
        tgt_pad_mask = tgt_words.data.eq(pad_idx).unsqueeze(1)  # [B, 1, T_tgt]

        with_align = kwargs.pop('with_align', False)
        attn_aligns = []

        for i, layer in enumerate(self.transformer_layers):
            layer_cache = self.state["cache"]["layer_{}".format(i)] \
                if step is not None else None
            output, attn, attn_align = layer(
                output,
                src_memory_bank,
                src_pad_mask,
                tgt_pad_mask,
                layer_cache=layer_cache,
                step=step,
                with_align=with_align)
            if attn_align is not None:
                attn_aligns.append(attn_align)

        output = self.layer_norm(output)
        dec_outs = output.transpose(0, 1).contiguous()
        attn = attn.transpose(0, 1).contiguous()

        attns = {"std": attn}
        if self._copy:
            attns["copy"] = attn
        if with_align:
            attns["align"] = attn_aligns[self.alignment_layer]  # `(B, Q, K)`
            # attns["align"] = torch.stack(attn_aligns, 0).mean(0)  # All avg

        # TODO change the way attns is returned dict => list or tuple (onnx)
        return dec_outs, attns

    def _init_cache(self, memory_bank):
        self.state["cache"] = {}
        batch_size = memory_bank.size(1)
        depth = memory_bank.size(-1)

        for i, layer in enumerate(self.transformer_layers):
            layer_cache = {"memory_keys": None, "memory_values": None}
            if isinstance(layer.self_attn, AverageAttention):
                layer_cache["prev_g"] = torch.zeros((batch_size, 1, depth),
                                                    device=memory_bank.device)
            else:
                layer_cache["self_keys"] = None
                layer_cache["self_values"] = None
            self.state["cache"]["layer_{}".format(i)] = layer_cache

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer_layers:
            layer.update_dropout(dropout, attention_dropout)
