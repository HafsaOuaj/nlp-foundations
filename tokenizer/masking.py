import torch


def create_causal_mask(seq_len: int) -> torch.Tensor:
    """Creates a causal mask for a sequence of given length.

    Args:
        seq_len (int): Length of the sequence.
    Returns:
        torch.Tensor: A causal mask of shape (seq_len, seq_len).
    """
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
    return mask


def create_padding_mask(batch,padding_token_id) -> torch.Tensor:
    """Creates a padding mask for attention mechanisms.

    Args:
        padding_mask (torch.Tensor): A tensor of shape (batch_size, seq_len)
create_causal_mask(2)
            where 1 indicates padding tokens and 0 indicates non-padding tokens.
    Returns:
        torch.Tensor: A padding mask of shape (batch_size, 1, 1, seq_len).
    """
    batch_size, seq_len = batch.shape
    padded = torch.zeros_like(batch).float().masked_fill(batch == padding_token_id, float('-inf'))
    mask = torch.zeros(batch_size, seq_len, seq_len)+ padded[:,:,None] + padded[:,None,:]
    return mask[:, None, :, :]


print(create_causal_mask(5))

batch = torch.tensor([
    [1, 2, 3, 4, 5, 6],
    [1, 2, 3, 0, 0, 0],
    [1, 2, 3, 4, 0, 0]
])
print(create_padding_mask(batch, 0))

import torch
 
def create_mask(query, key, padding_token_id):
    """
    Create a padding mask for a batch of sequences.
 
    Args:
        query: Batch of sequences for query, shape (batch_size, query_len)
        key: Batch of sequences for key, shape (batch_size, key_len)
        padding_token_id: ID of the padding token
 
    Returns:
        Padding mask of shape (batch_size, query_len, key_len)
    """
    batch_size, query_len = query.shape
    _, key_len = key.shape
    q_padded = torch.zeros_like(query).float().masked_fill(query == padding_token_id, float('-inf'))
    k_padded = torch.zeros_like(key).float().masked_fill(key == padding_token_id, float('-inf'))
    mask = torch.zeros(batch_size, query_len, key_len) + q_padded[:,:,None] + k_padded[:,None,:]
    return mask

dim = 16
num_heads = 4
attn_layer = torch.nn.MultiheadAttention(dim, num_heads, dropout=0.1, batch_first=True)

# Input tensor: 0 = padding
batch = torch.tensor([
    [1, 2, 3, 4, 5, 6],
    [1, 2, 3, 0, 0, 0],
    [1, 2, 3, 4, 0, 0]
])
batch_size, seq_len = batch.shape
x = torch.randn(batch_size, seq_len, dim)

attn_mask = create_mask(batch, batch, 0)
attn_mask = attn_mask.repeat(1, num_heads, 1, 1).view(-1, seq_len, seq_len)

y = attn_layer(x, x, x, key_padding_mask=None, attn_mask=attn_mask)
 