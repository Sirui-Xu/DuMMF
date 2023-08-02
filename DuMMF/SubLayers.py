''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from DuMMF.Modules import ScaledDotProductAttention
import torch
import numpy as np
from torch import Tensor
from typing import Optional, Any, Union, Callable
import torch.nn.functional as F
import math
from local_attention import LocalAttention
from torchvision.ops import stochastic_depth


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head # 64, 64, 8
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        # print(k.shape)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        # print(k.shape)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)
        
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn




class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

def _normalize_and_reshape_query(q, heads, unit_norm, depth_scale, normalize_stop_grads=False):
    """Normalizes the query and prepares it for attention computation."""
    newshape = [heads, q.shape[-1] // heads]
    newshape = [*q.shape[:-1], *newshape]
    q = q.reshape(newshape)
    if unit_norm:
        if normalize_stop_grads:
            with torch.no_grad():
                q_norm = torch.norm(q, dim=-1, keepdim=True)
        else:
            q_norm = torch.norm(q, dim=-1, keepdim=True)
        q = q / (q_norm + 1e-6)
    if depth_scale:
        depth = q.shape[-1]
        q = q / math.sqrt(depth)
    newshape = [*q.shape[:-2], -1]
    q = q.reshape(newshape)
    return q

class TransformerEncoderLayerQaNV0(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1, num_queries: int = 10, num_persons: int = 3, window_size: int = 1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayerQaNV0, self).__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
        #                                     **factory_kwargs)

        self.self_attn = LocalAttention(
            dim = d_model,           # dimension of each head (you need to pass this in for relative positional encoding)
            window_size = window_size,       # window size. 512 is optimal, but 256 or 128 yields good enough results
            causal = False,           # auto-regressive or not
            look_backward = 1,       # each window looks at the window before
            look_forward = 1,        # for non-auto-regressive case, will default to 1, so each window looks at the window before and after it
            dropout = dropout,           # post-attention dropout
            exact_windowsize = False, # if this is set to true, in the causal setting, each query will see at maximum the number of keys equal to the window size
            autopad = True
        )

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.queries = nn.Parameter(torch.FloatTensor(num_queries, d_model)) 
        stdv = 1. / math.sqrt(self.queries.size(1))
        self.queries.data.normal_(-stdv,stdv)

        self.wk = nn.Parameter(torch.FloatTensor(num_queries, 1)) 
        stdv = 1. / math.sqrt(self.wk.size(0))
        self.wk.data.normal_(-stdv,stdv)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

        self.d_model = d_model
        self.nhead = nhead
        self.num_queries = num_queries
        self.dropout_rate = 0

    def _get_query(self, x):
        """Defines the queries parameters and reshape then into [..., N_Queries, Heads, Head_dim].
        Queries are also scaled by SQRT(Head_dim) for stability (proposed in https://arxiv.org/abs/1706.03762).
        Returns: Queries after preprocessing with shape [..., N_Queries, Heads, Head_dim].
        """
        q = _normalize_and_reshape_query(self.queries, self.nhead, True, depth_scale=True
                                        )
        # B, N, T, d
        q = q.unsqueeze(0).unsqueeze(2).repeat(x.shape[1], 1, x.shape[0], 1).contiguous()
        return q

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayerQaN, self).__setstate__(state)

    def forward(self, src: Tensor, memory: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, global_level: float = 0.0) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src.clone()
        if self.norm_first:
            x = x + self._qa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._qa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        residual = stochastic_depth((x - src).permute(1, 0, 2), self.dropout_rate, 'row', self.training).permute(1, 0, 2).contiguous()
        x = src + residual

        return x

    # def _compute_QK_scores(self, q, x):
    #     """Computes the QK dot product in fused manner.
    #     Since the queries are shared across windows, we compute (Q*W_k^T)X^T for better memory utilization.
    #     :param q: The leared queries of shape [..., N_Queries, Heads, Head_dim]
    #     :param x: The input features [B, H, W, C]
    #     :return: The query-key dot product for each query head [B, H, W, N_Queries, Heads]
    #     """
    #     # q = [Nq, h, d]
    #     # WK = [D_in, h, D]
    #     T, B, D = x.shape
    #     q = q.unsqueeze(0).repeat(B, 1, 1, 1)
    #     Wk = Wk.reshape([-1, self.nhead, self.d_model // self.nhead])
    #     qWk = torch.einsum('Bqhd,Dhd->BDqh', q, Wk)
    #     qWkx = torch.einsum('TBD,BDqh->TBqh', x, qWk)

    #     return qWkx

    def _qa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        q = self._get_query(x)
        # T B D -> N T B D -> B N T D
        x = x.unsqueeze(0).repeat(self.num_queries, 1, 1, 1).permute(2, 0, 1, 3).contiguous()
        B, N, T, D = x.shape
        mask = torch.ones(1, T).bool().to(x.device)
        x = self.self_attn(q.view(B * N, T, D), x.view(B * N, T, D), x.view(B * N, T, D), mask = mask).view(B, N, T, D)
        x = torch.einsum("bntd,nk->bktd", (x, self.wk)).squeeze(1).permute(1, 0, 2).contiguous()
        return self.dropout1(x)

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        T, B, D = x.shape
        mask = torch.ones(1, T).bool().to(x.device)
        x = x.permute(1, 0, 2).contiguous()
        x = self.self_attn(x, x, x, mask = mask).permute(1, 0, 2).contiguous()
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
    

class TransformerEncoderLayerQaN(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1, num_queries: int = 10, num_persons: int = 3, window_size: int = 1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayerQaN, self).__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
        #                                     **factory_kwargs)

        self.self_attn = LocalAttention(
            dim = d_model,           # dimension of each head (you need to pass this in for relative positional encoding)
            window_size = window_size,       # window size. 512 is optimal, but 256 or 128 yields good enough results
            causal = False,           # auto-regressive or not
            look_backward = 1,       # each window looks at the window before
            look_forward = 1,        # for non-auto-regressive case, will default to 1, so each window looks at the window before and after it
            dropout = dropout,           # post-attention dropout
            exact_windowsize = False, # if this is set to true, in the causal setting, each query will see at maximum the number of keys equal to the window size
            autopad = True
        )

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.queries = nn.Parameter(torch.FloatTensor(num_queries, d_model)) 
        stdv = 1. / math.sqrt(self.queries.size(1))
        self.queries.data.normal_(-stdv,stdv)

        # self.wk = nn.Parameter(torch.FloatTensor(num_queries, 1)) 
        # stdv = 1. / math.sqrt(self.wk.size(0))
        # self.wk.data.normal_(-stdv,stdv)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.multi_attn_global = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                **factory_kwargs)
        
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout3 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

        self.d_model = d_model
        self.nhead = nhead
        self.num_queries = num_queries
        self.num_persons = num_persons
        self.dropout_rate = 0

    def _get_query(self, x):
        """Defines the queries parameters and reshape then into [..., N_Queries, Heads, Head_dim].
        Queries are also scaled by SQRT(Head_dim) for stability (proposed in https://arxiv.org/abs/1706.03762).
        Returns: Queries after preprocessing with shape [..., N_Queries, Heads, Head_dim].
        """
        q = _normalize_and_reshape_query(self.queries, self.nhead, True, depth_scale=True
                                        )
        # B, N, T, d
        q = q.unsqueeze(0).unsqueeze(2).repeat(x.shape[1] // self.num_queries, 1, x.shape[0], 1).contiguous()

        return q

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayerQaN, self).__setstate__(state)

    def forward(self, src: Tensor, memory: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, global_level: float = 0.0) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src.clone()
        if self.norm_first:
            x = x + self._qa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + global_level * self._mm_block(self.norm3(x), memory, src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))

        else:
            x = self.norm1(x + self._qa_block(x, src_mask, src_key_padding_mask))
            x = self.norm3(x + global_level * self._mm_block(x, memory, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        # residual = stochastic_depth((x - src).permute(1, 0, 2), self.dropout_rate, 'row', self.training).permute(1, 0, 2).contiguous()
        # x = src + residual

        return x

    def _qa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        q = self._get_query(x)
        # T BNM D -> T B N M D -> BM N T D
        T, _, D = x.shape
        x = x.view(T, -1, self.num_queries, self.num_persons, D).permute(1, 3, 2, 0, 4).contiguous().view(-1, self.num_queries, T, D)
        # x = x.unsqueeze(0).repeat(self.num_queries, 1, 1, 1).permute(2, 0, 1, 3).contiguous()
        BM, N, _, D = x.shape
        mask = torch.ones(1, T).bool().to(x.device)
        x = self.self_attn(q.view(BM * N, T, D), x.view(BM * N, T, D), x.view(BM * N, T, D), mask = mask).view(BM, N, T, D)
        # x = torch.einsum("bntd,nk->bktd", (x, self.wk)).squeeze(1).permute(1, 0, 2).contiguous()
        x = x.view(-1, self.num_persons, N, T, D).permute(3, 0, 2, 1, 4).contiguous().view(T, -1, D)
        return self.dropout1(x)

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        T, B, D = x.shape
        mask = torch.ones(1, T).bool().to(x.device)
        x = x.permute(1, 0, 2).contiguous()
        x = self.self_attn(x, x, x, mask = mask).permute(1, 0, 2).contiguous()
        return self.dropout1(x)

    def _mm_block(self, x: Tensor, memory: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        
        # T BNM D -> T B N M D -> M TBN D
        T, _, D = x.shape
        x = x.view(T, -1, self.num_queries, self.num_persons, D).permute(3, 0, 1, 2, 4).contiguous().view(self.num_persons, -1, D)
        memory = memory.view(T, -1, self.num_queries, self.num_persons, D).permute(3, 0, 1, 2, 4).contiguous().view(self.num_persons, -1, D)
        x = self.multi_attn_global(x+memory, x+memory, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        x = x.view(self.num_persons, T, -1, self.num_queries, D).permute(1, 2, 3, 0, 4).contiguous().view(T, -1, D)
        return self.dropout3(x)
    
    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerDecoderLayerQaN(nn.Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectivaly. Otherwise it's done after.
            Default: ``False`` (after).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)

    Alternatively, when ``batch_first`` is ``True``:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> memory = torch.rand(32, 10, 512)
        >>> tgt = torch.rand(32, 20, 512)
        >>> out = decoder_layer(tgt, memory)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1, num_queries: int = 10, num_persons: int = 3, window_size: int = 1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerDecoderLayerQaN, self).__init__()
                # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
        #                                     **factory_kwargs)

        self.self_attn = LocalAttention(
            dim = d_model,           # dimension of each head (you need to pass this in for relative positional encoding)
            window_size = window_size,       # window size. 512 is optimal, but 256 or 128 yields good enough results
            causal = False,           # auto-regressive or not
            look_backward = 1,       # each window looks at the window before
            look_forward = 1,        # for non-auto-regressive case, will default to 1, so each window looks at the window before and after it
            dropout = dropout,           # post-attention dropout
            exact_windowsize = False, # if this is set to true, in the causal setting, each query will see at maximum the number of keys equal to the window size
            autopad = True
        )
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.queries = nn.Parameter(torch.FloatTensor(num_queries, d_model)) 
        stdv = 1. / math.sqrt(self.queries.size(1))
        self.queries.data.normal_(-stdv,stdv)

        # self.wk = nn.Parameter(torch.FloatTensor(num_queries, 1)) 
        # stdv = 1. / math.sqrt(self.wk.size(0))
        # self.wk.data.normal_(-stdv,stdv)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.multi_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                **factory_kwargs)
        self.norm4 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout4 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

        self.d_model = d_model
        self.nhead = nhead
        self.num_queries = num_queries
        self.num_persons = num_persons
        self.dropout_rate = 0

    def _get_query(self, x):
        """Defines the queries parameters and reshape then into [..., N_Queries, Heads, Head_dim].
        Queries are also scaled by SQRT(Head_dim) for stability (proposed in https://arxiv.org/abs/1706.03762).
        Returns: Queries after preprocessing with shape [..., N_Queries, Heads, Head_dim].
        """
        q = _normalize_and_reshape_query(self.queries, self.nhead, True, depth_scale=True
                                        )
        # B, N, T, d
        q = q.unsqueeze(0).unsqueeze(2).repeat(x.shape[1] // self.num_queries, 1, x.shape[0], 1).contiguous()
        return q

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayerQaN, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None, global_level: float = 0.0) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt.clone()
        if self.norm_first:
            x = x + self._qa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + global_level * self._mm_block(self.norm4(x), tgt_mask, tgt_key_padding_mask)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._qa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm4(x + global_level * self._mm_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))

        # residual = stochastic_depth((x - tgt).permute(1, 0, 2), self.dropout_rate, 'row', self.training).permute(1, 0, 2).contiguous()
        # x = tgt + residual

        return x

    def _qa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        q = self._get_query(x)
        # T BNM D -> T B N M D -> BM N T D
        T, _, D = x.shape
        x = x.view(T, -1, self.num_queries, self.num_persons, D).permute(1, 3, 2, 0, 4).contiguous().view(-1, self.num_queries, T, D)
        # x = x.unsqueeze(0).repeat(self.num_queries, 1, 1, 1).permute(2, 0, 1, 3).contiguous()
        BM, N, _, D = x.shape
        mask = torch.ones(1, T).bool().to(x.device)
        x = self.self_attn(q.view(BM * N, T, D), x.view(BM * N, T, D), x.view(BM * N, T, D), mask = mask).view(BM, N, T, D)
        # x = torch.einsum("bntd,nk->bktd", (x, self.wk)).squeeze(1).permute(1, 0, 2).contiguous()
        x = x.view(-1, self.num_persons, N, T, D).permute(3, 0, 2, 1, 4).contiguous().view(T, -1, D)
        return self.dropout1(x)

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        T, B, D = x.shape
        mask = torch.ones(1, T).bool().to(x.device)
        x = x.permute(1, 0, 2).contiguous()
        x = self.self_attn(x, x, x, mask = mask).permute(1, 0, 2).contiguous()
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

    def _mm_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        
        # T BNM D -> T B N M D -> M TBN D
        T, _, D = x.shape
        x = x.view(T, -1, self.num_queries, self.num_persons, D).permute(3, 0, 1, 2, 4).contiguous().view(self.num_persons, -1, D)
        x = self.multi_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        x = x.view(self.num_persons, T, -1, self.num_queries, D).permute(1, 2, 3, 0, 4).contiguous().view(T, -1, D)
        return self.dropout4(x)
    
    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)
    


class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, memory: Optional[Tensor] = None, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, global_level: bool = True) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerDecoderLayer(nn.Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectivaly. Otherwise it's done after.
            Default: ``False`` (after).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)

    Alternatively, when ``batch_first`` is ``True``:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> memory = torch.rand(32, 10, 512)
        >>> tgt = torch.rand(32, 20, 512)
        >>> out = decoder_layer(tgt, memory)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None, global_level: bool = True) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)