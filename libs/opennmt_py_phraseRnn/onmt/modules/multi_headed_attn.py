""" Multi-Head Attention module """
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from onmt.utils.misc import generate_relative_positions_matrix,\
                            relative_matmul
# from onmt.utils.misc import aeq


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
        return out/self.n_gram


class NgramLSTM(nn.Module):
    def __init__(self, n_gram, input_size):
        super(NgramLSTM, self).__init__()
        self.n_gram = n_gram

        self._num_layers = 1
        self.input_size = input_size
        self.hidden_size = input_size

        self.rnn = nn.RNN(self.input_size,
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
        _bank_mt, _h_n = self.rnn(zz)
        # _aggregate_hidden_n = torch.cat((_h_n, c_n), dim=0)
        out = torch.sum(_h_n, dim=0)

        # finally, we reshape original batch_size to return
        # (batch x seq x hidden_size)
        out = out.reshape(batch_size, -1, hidden_size)
        return out


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention module from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.

    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.

    .. mermaid::

       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O

    Also includes several additional tricks.

    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, head_count, model_dim, dropout=0.1,
                 max_relative_positions=0, gram_sizes=None):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)

        self.max_relative_positions = max_relative_positions

        self.n_gram_features = None
        if gram_sizes is not None and len(gram_sizes) == head_count:
            ngram_size_info = dict([("{}_gram_features".format(gram_size), NgramLSTM(gram_size, self.dim_per_head))
                                    for gram_size in set(gram_sizes) if gram_size > 0])
            self.n_gram_features = nn.ModuleDict(ngram_size_info)
            self.n_gram_features_count = dict([(gram_size, len([_x for _x in gram_sizes if _x == gram_size]))
                                               for gram_size in set(gram_sizes)])

        if max_relative_positions > 0:
            vocab_size = max_relative_positions * 2 + 1
            self.relative_positions_embeddings = nn.Embedding(
                vocab_size, self.dim_per_head)

    def forward(self, key, value, query, mask=None,
                layer_cache=None, attn_type=None):
        """
        Compute the context vector and the attention vectors.

        Args:
           key (FloatTensor): set of `key_len`
               key vectors ``(batch, key_len, dim)``
           value (FloatTensor): set of `key_len`
               value vectors ``(batch, key_len, dim)``
           query (FloatTensor): set of `query_len`
               query vectors  ``(batch, query_len, dim)``
           mask: binary mask 1/0 indicating which keys have
               zero / non-zero attention ``(batch, query_len, key_len)``
        Returns:
           (FloatTensor, FloatTensor):

           * output context vectors ``(batch, query_len, dim)``
           * Attention vector in heads ``(batch, head, query_len, key_len)``.
        """

        # CHECKS
        # batch, k_len, d = key.size()
        # batch_, k_len_, d_ = value.size()
        # aeq(batch, batch_)
        # aeq(k_len, k_len_)
        # aeq(d, d_)
        # batch_, q_len, d_ = query.size()
        # aeq(batch, batch_)
        # aeq(d, d_)
        # aeq(self.model_dim % 8, 0)
        # if mask is not None:
        #    batch_, q_len_, k_len_ = mask.size()
        #    aeq(batch_, batch)
        #    aeq(k_len_, k_len)
        #    aeq(q_len_ == q_len)
        # END CHECKS

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            """Projection."""
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """Compute context."""
            return x.transpose(1, 2).contiguous() \
                    .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if attn_type == "self":
                query, key, value = self.linear_query(query),\
                                    self.linear_keys(query),\
                                    self.linear_values(query)
                key = shape(key)
                value = shape(value)
                if layer_cache["self_keys"] is not None:
                    key = torch.cat(
                        (layer_cache["self_keys"], key),
                        dim=2)
                if layer_cache["self_values"] is not None:
                    value = torch.cat(
                        (layer_cache["self_values"], value),
                        dim=2)
                layer_cache["self_keys"] = key
                layer_cache["self_values"] = value
            elif attn_type == "context":
                query = self.linear_query(query)
                if layer_cache["memory_keys"] is None:
                    key, value = self.linear_keys(key),\
                                 self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
                else:
                    key, value = layer_cache["memory_keys"],\
                               layer_cache["memory_values"]
                layer_cache["memory_keys"] = key
                layer_cache["memory_values"] = value
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)

        if self.max_relative_positions > 0 and attn_type == "self":
            key_len = key.size(2)
            # 1 or key_len x key_len
            relative_positions_matrix = generate_relative_positions_matrix(
                key_len, self.max_relative_positions,
                cache=True if layer_cache is not None else False)
            #  1 or key_len x key_len x dim_per_head
            relations_keys = self.relative_positions_embeddings(
                relative_positions_matrix.to(key.device))
            #  1 or key_len x key_len x dim_per_head
            relations_values = self.relative_positions_embeddings(
                relative_positions_matrix.to(key.device))

        query = shape(query)

        # ngram feature for q, k, v
        if self.n_gram_features is not None:
            if mask is not None:
                mask_qkv = mask.unsqueeze(-1)  # [B, 1, seq_len, 1]
                query = query.masked_fill(mask_qkv, 0)
                key = key.masked_fill(mask_qkv, 0)
                value = value.masked_fill(mask_qkv, 0)

            idx_head_layer = 0
            for gram_size, count_h_using in self.n_gram_features_count.items():
                if gram_size == 0:
                    idx_head_layer += count_h_using
                    continue
                ngram_features_extractor = self.n_gram_features["{}_gram_features".format(gram_size)]
                _xx = torch.cat([ query[:, idx_head_layer:idx_head_layer+count_h_using, :, :].reshape(-1, query_len, dim_per_head),
                                  key[:, idx_head_layer:idx_head_layer+count_h_using, :, :].reshape(-1, query_len, dim_per_head),
                                  value[:, idx_head_layer:idx_head_layer+count_h_using, :, :].reshape(-1, query_len, dim_per_head)
                                  ], dim=0).reshape(-1, query_len, dim_per_head)
                _yy = ngram_features_extractor(_xx).reshape(3, -1, query_len, dim_per_head)
                _q, _k, _v = _yy[0], _yy[1], _yy[2]
                query[:, idx_head_layer:idx_head_layer+count_h_using, :, :] = _q.reshape(batch_size, -1, query_len, dim_per_head)
                key[:, idx_head_layer:idx_head_layer+count_h_using, :, :] = _k.reshape(batch_size, -1, query_len, dim_per_head)
                value[:, idx_head_layer:idx_head_layer+count_h_using, :, :] = _v.reshape(batch_size, -1, query_len, dim_per_head)
                idx_head_layer += count_h_using

        key_len = key.size(2)
        query_len = query.size(2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        # batch x num_heads x query_len x key_len
        query_key = torch.matmul(query, key.transpose(2, 3))

        if self.max_relative_positions > 0 and attn_type == "self":
            scores = query_key + relative_matmul(query, relations_keys, True)
        else:
            scores = query_key
        scores = scores.float()

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores).to(query.dtype)
        drop_attn = self.dropout(attn)

        context_original = torch.matmul(drop_attn, value)

        if self.max_relative_positions > 0 and attn_type == "self":
            context = unshape(context_original
                              + relative_matmul(drop_attn,
                                                relations_values,
                                                False))
        else:
            context = unshape(context_original)

        output = self.final_linear(context)
        # CHECK
        # batch_, q_len_, d_ = output.size()
        # aeq(q_len, q_len_)
        # aeq(batch, batch_)
        # aeq(d, d_)

        # Return multi-head attn
        attns = attn \
            .view(batch_size, head_count,
                  query_len, key_len)

        return output, attns

    def update_dropout(self, dropout):
        self.dropout.p = dropout
