# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, Optional, Tuple
from fairseq.modules.multihead_attention import MultiheadAttention

import torch
import torch.nn.functional as F
from fairseq import utils
from torch import Tensor, nn
import numpy as np

class NgramCombined(nn.Module):
    def __init__(self, n_gram):
        super(NgramCombined, self).__init__()
        self.n_gram = n_gram

    def forward(self, x):
        out = x
        if self.n_gram > 1:
            for i_gram in range(1, self.n_gram):
                out = F.pad(x.transpose(-1, -2), [i_gram, 0],
                            mode='constant', value=0).transpose(-1, -2)[:,:-i_gram,:] + out
        return out / self.n_gram



class NgramMinPooling(nn.Module):
    def __init__(self, n_gram, input_size, phrase_drop=0.3):
        super(NgramMinPooling, self).__init__()
        self.n_gram = n_gram

        self.phrase_drop = phrase_drop
        self.dropout = nn.Dropout(0.3) 

    def forward(self, _x):
        # we need to create a new data input to learn the n-gram (k) feature using LSTM
        # with origin input (_x) = [emb_1, emb_2, emb_3 .. emb_{seq_length}]: batchsize x seq_length x emb_size
        n_gram = self.n_gram
        data_input = _x.unsqueeze(dim=0)
        data_org = _x

        batch_size = data_org.size(0)
        seq_length = data_org.size(1)
        hidden_size = data_org.size(-1)

        #
        # 1. add padding k - 1 times =>  [k x batch_size x seq_length x emb_size]
        #    [zero_1, .. zero_{k-1}, emb_1, emb_2, emb_3 .. emb_{seq_length - k + 1}]: batchsize x seq_length x emb_size
        #    [zero_1, .. emb_1,      emb_2, emb_3 ..        emb_{seq_length - k + 2}]: batchsize x seq_length x emb_size
        #    ...
        #    [emb_1, emb_2, emb_3 ..                        emb_{seq_length}]: batchsize x seq_length x emb_size
        for i_gram in range(1, n_gram):
            mt_padd_i = F.pad(data_org.transpose(-1,-2), [i_gram, 0],
                              mode='constant', value=0).transpose(-1,-2)[:,:-i_gram,:]
            data_input = torch.cat((mt_padd_i.unsqueeze(dim=0), data_input), dim=0)

            #
        rand_index = np.random.permutation(batch_size * seq_length)[:int(batch_size * seq_length * (1 - self.phrase_drop))]
        rand_index.sort()
        rand_index = torch.LongTensor(rand_index).to(device=data_input.device)

         # reshape input into =>   [(batch_size x seq_length) x k x emb_size]
        # this mean that we cut the sentence into many sentence piece (k-gram) similar
        # n-gram in NLP, and combined all set of n-gram treat to LSTM as a batch of input
        zz = data_input.view(n_gram, -1, hidden_size).index_select(1, rand_index)

        # forward data using min value
        out, _ = torch.min(self.dropout(zz), dim=0)

        # copy back phrase states override word states 
        out = _x.view(-1, hidden_size).index_copy(0, rand_index, out)

        # finally, we reshape original batch_size to return
        # (batch x seq x hidden_size)
        out = out.reshape(batch_size, -1, hidden_size)

        rate_local_context = torch.sigmoid(_x)
        out = rate_local_context*out + (1 - rate_local_context)*_x

        return out


class NgramLSTM(nn.Module):
    def __init__(self, n_gram, input_size):
        super(NgramLSTM, self).__init__()
        self.n_gram = n_gram

        self._num_layers = 1
        self.input_size = input_size
        self.hidden_size = input_size

        self.dropout = nn.Dropout(0.3)
        self.rnn = nn.LSTM(self.input_size,
                           self.hidden_size,
                           batch_first=False,
                           num_layers=self._num_layers,
                           bidirectional=True)

    def forward(self, _x):
        # we need to create a new data input to learn the n-gram (k) feature using LSTM
        # with origin input (_x) = [emb_1, emb_2, emb_3 .. emb_{seq_length}]: batchsize x seq_length x emb_size
        n_gram = self.n_gram
        data_input = _x.unsqueeze(dim=0)
        data_org = _x

        batch_size = data_org.size(0)
        seq_length = data_org.size(1)
        hidden_size = self.hidden_size
        input_size = self.input_size

        #
        # 1. add padding k - 1 times =>  [k x batch_size x seq_length x emb_size]
        #    [zero_1, .. zero_{k-1}, emb_1, emb_2, emb_3 .. emb_{seq_length - k + 1}]: batchsize x seq_length x emb_size
        #    [zero_1, .. emb_1,      emb_2, emb_3 ..        emb_{seq_length - k + 2}]: batchsize x seq_length x emb_size
        #    ...
        #    [emb_1, emb_2, emb_3 ..                        emb_{seq_length}]: batchsize x seq_length x emb_size
        for i_gram in range(1, n_gram):
            mt_padd_i = F.pad(data_org.transpose(-1,-2), [i_gram, 0],
                              mode='constant', value=0).transpose(-1,-2)[:,:-i_gram,:]
            data_input = torch.cat((mt_padd_i.unsqueeze(dim=0), data_input), dim=0)

            #
        # reshape input into =>   [(batch_size x seq_length) x k x emb_size]
        # this mean that we cut the sentence into many sentence piece (k-gram) similar
        # n-gram in NLP, and combined all set of n-gram treat to LSTM as a batch of input
        zz = data_input.reshape([n_gram,
                                 batch_size * seq_length,
                                 hidden_size])

        # forward data using Bi-LSTM
        # we just get the cell state (num_layers * num_directions, batch, hidden_size)
        # because we need to get the long-memmory to extract the k-gram features of words
        # in this case, we use num_layers = 1, num_directions=2,
        # we sum all directions
        _bank_mt, (_h_n, c_n) = self.rnn(self.dropout(zz))
        out = torch.sum(_h_n+c_n, dim=0)

        # finally, we reshape original batch_size to return
        # (batch x seq x hidden_size)

        out = out.reshape(batch_size, -1, hidden_size)

        # rate_local_context = torch.sigmoid(_x)
        # out = rate_local_context*out + (1 - rate_local_context)*_x
        return out

class PhraseMultiheadAttention(MultiheadAttention):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, *args, **kwargs):
        self.ngram_sizes=kwargs.get("ngram_sizes", None)
        self.phrase_dropout=kwargs.get("phrase_dropout", None)
        self.num_phrase_layers=kwargs.get("num_phrase_layers", None)
        kwargs.pop("ngram_sizes")
        kwargs.pop("phrase_dropout")
        kwargs.pop("num_phrase_layers")
        
        super().__init__(*args, **kwargs)

        if self.ngram_sizes is not None: 
            assert (
                len(self.ngram_sizes) == self.num_heads
            ), "gram_sizes need be setup for all heads: self.ngram_sizes={}, self.num_heads={}".format(self.ngram_sizes, self.num_heads)
            ngram_size_info = dict([("{}_gram_features".format(gram_size), NgramLSTM(gram_size, self.head_dim))
                                    for gram_size in set(self.ngram_sizes) if gram_size > 0])
            self.n_gram_features = nn.ModuleDict(ngram_size_info)
            self.n_gram_features_count = dict([(gram_size, len([_x for _x in self.ngram_sizes if _x == gram_size]))
                                            for gram_size in set(self.ngram_sizes)])
            self.init_parameters()

    def init_parameters(self):
        for gram_size, _ in self.n_gram_features_count.items():
            if gram_size == 0:
                continue
            ngram_features_extractor = self.n_gram_features["{}_gram_features".format(gram_size)]
            for name, param in ngram_features_extractor.named_parameters():
                if 'bias' in name or 'weight' in name:
                    if len(param.shape) > 1:
                        nn.init.xavier_uniform_(param) 

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        # if (
        #     not self.onnx_trace
        #     and not self.tpu  # don't use PyTorch version on TPUs
        #     and incremental_state is None
        #     and not static_kv
        #     # A workaround for quantization to work. Otherwise JIT compilation
        #     # treats bias in linear module as method.
        #     and not torch.jit.is_scripting()
        # ):
        #     assert key is not None and value is not None
        #     return F.multi_head_attention_forward(
        #         query,
        #         key,
        #         value,
        #         self.embed_dim,
        #         self.num_heads,
        #         torch.empty([0]),
        #         torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
        #         self.bias_k,
        #         self.bias_v,
        #         self.add_zero_attn,
        #         self.dropout_module.p,
        #         self.out_proj.weight,
        #         self.out_proj.bias,
        #         self.training or self.dropout_module.apply_during_inference,
        #         key_padding_mask,
        #         need_weights,
        #         attn_mask,
        #         use_separate_proj_weight=True,
        #         q_proj_weight=self.q_proj.weight,
        #         k_proj_weight=self.k_proj.weight,
        #         v_proj_weight=self.v_proj.weight,
        #     )

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)

        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(
                            key_padding_mask
                        ),
                    ],
                    dim=1,
                )

        # ==========================
        # phrase mechanism 
        # ngram feature for q, k, v
        # ==========================
        if self.n_gram_features is not None:
            if key_padding_mask is not None:
                mask_qkv = key_padding_mask.unsqueeze(-1).unsqueeze(1)  # [B, 1, seq_len, 1] masked for [bsz, num_heads, seq_len, head_dim]
                _q0 = q.view(bsz, self.num_heads, -1, self.head_dim).masked_fill(mask_qkv, 0)
                _k0 = k.view(bsz, self.num_heads, -1, self.head_dim).masked_fill(mask_qkv, 0)
                _v0 = v.view(bsz, self.num_heads, -1, self.head_dim).masked_fill(mask_qkv, 0)

            idx_head_layer = 0
            for gram_size, count_h_using in self.n_gram_features_count.items():
                if gram_size == 0:
                    idx_head_layer += count_h_using
                    continue
                ngram_features_extractor = self.n_gram_features["{}_gram_features".format(gram_size)]
                _xx = torch.cat([ _q0.view(bsz, self.num_heads, -1, self.head_dim)[:, idx_head_layer:idx_head_layer+count_h_using, :, :].reshape(-1, src_len, self.head_dim),
                                  _k0.view(bsz, self.num_heads, -1, self.head_dim)[:, idx_head_layer:idx_head_layer+count_h_using, :, :].reshape(-1, tgt_len, self.head_dim),
                                  _v0.view(bsz, self.num_heads, -1, self.head_dim)[:, idx_head_layer:idx_head_layer+count_h_using, :, :].reshape(-1, tgt_len, self.head_dim)
                                  ], dim=0).reshape(-1, src_len, self.head_dim)
                _yy = ngram_features_extractor(_xx).reshape(3, -1, src_len, self.head_dim)
                _q, _k, _v = _yy[0], _yy[1], _yy[2]
                q.view(bsz, self.num_heads, -1, self.head_dim)[:, idx_head_layer:idx_head_layer+count_h_using, :, :] = _q.reshape(bsz, -1, src_len, self.head_dim)
                k.view(bsz, self.num_heads, -1, self.head_dim)[:, idx_head_layer:idx_head_layer+count_h_using, :, :] = _k.reshape(bsz, -1, tgt_len, self.head_dim)
                v.view(bsz, self.num_heads, -1, self.head_dim)[:, idx_head_layer:idx_head_layer+count_h_using, :, :] = _v.reshape(bsz, -1, tgt_len, self.head_dim)
                
                idx_head_layer += count_h_using
            
            if key_padding_mask is not None:
                mask_qkv = key_padding_mask.unsqueeze(-1).unsqueeze(1)  # [B, 1, seq_len, 1] masked for [bsz, num_heads, seq_len, head_dim]
                q = q.view(bsz, self.num_heads, -1, self.head_dim).masked_fill(mask_qkv, 0).view(bsz*self.num_heads, -1, self.head_dim)
                k = k.view(bsz, self.num_heads, -1, self.head_dim).masked_fill(mask_qkv, 0).view(bsz*self.num_heads, -1, self.head_dim)
                v = v.view(bsz, self.num_heads, -1, self.head_dim).masked_fill(mask_qkv, 0).view(bsz*self.num_heads, -1, self.head_dim)

        # ==========================

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not self.tpu:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights
 