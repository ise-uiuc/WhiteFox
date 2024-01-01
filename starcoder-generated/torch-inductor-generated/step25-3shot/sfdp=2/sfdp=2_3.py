
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, num_heads, embed_dim, dropout = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.all_head_dim = self.head_dim * num_heads
        self.embed_dim = embed_dim
        self.dropout_p = dropout
        self.query = torch.nn.Linear(embed_dim, self.all_head_dim)
        self.key = torch.nn.Linear(embed_dim, self.all_head_dim)
        self.value = torch.nn.Linear(embed_dim, self.all_head_dim)
        self.out = torch.nn.Linear(embed_dim, self.all_head_dim)
 
    def forward(self, query, key, value, key_padding_mask = None, need_weights = True):
        mixed_query = self.query(query)
        mixed_key = self.key(key)
        mixed_value = self.value(value)
        query_shape = query.shape[:-1]
        key_shape = key.shape[:-1]
        mixed_shape = mixed_query.shape[:-1]
        assert mixed_shape == query_shape and mixed_shape == key_shape, f"mixed_shape: {mixed_shape}, query_shape: {query_shape}, key_shape: {query_shape}"
        query_group = mixed_query.view(*mixed_shape, self.num_heads, self.head_dim)
        key_group = mixed_key.view(*mixed_shape, self.num_heads, self.head_dim)
        value_group = mixed_value.view(*mixed_shape, self.num_heads, self.head_dim)
        qk = torch.matmul(query_group, key_group.transpose(-2, -1))
        qk_scale = qk.div(self.head_dim ** 0.5)
        if key_padding_mask is not None:
            qk_scale.masked_fill_(key_padding_mask, float('-inf'))
        softmax_qk = torch.nn.functional.softmax(qk_scale, dim=-1)
        softmax_qk_dropout = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = softmax_qk_dropout.matmul(value_group)
        output = output.contiguous().view(*mixed_shape, self.all_head_dim)
        return self.out(output)

# Initializing the model
m = MultiHeadAttention(8, 32)

# Inputs to the model
query = torch.randn(1, 8, 32)
key = torch.randn(1, 16, 32)
value = torch.randn(1, 16, 32)
key_padding_mask = torch.tensor([[0, 0, 0, 1, 1, 0, 0]])
