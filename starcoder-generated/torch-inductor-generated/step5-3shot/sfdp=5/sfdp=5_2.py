
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_dim, num_heads, dropout_p=0, bias=True, add_bias_kv=False, add_zero_attn=False):
            super().__init__()
            self.query = LinearWeightNorm(input_dim, num_heads * input_dim, bias=bias)
            self.key = LinearWeightNorm(input_dim, num_heads * input_dim, bias=bias)
            self.value = LinearWeightNorm(input_dim, num_heads * input_dim, bias=bias)
            self.head_dim = input_dim // num_heads
            self.scale = self.head_dim ** -0.5
            self.norm = BatchedLayerNormal(input_dim)
            self.attn_dropout = nn.Dropout(dropout_p)
            self.proj_dropout = nn.Dropout(dropout_p)
 
    def forward(self, query, key, value, attn_mask=None):
        # Check input shapes
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.head_dim * self.head_dim, "The embedded dimension of query does not equal to the product of number of heads and number of outputs per head."
        src_len, bsz_, embed_dim = key.size()
        assert embed_dim == self.head_dim * self.head_dim, "The embedded dimension of key does not equal to the product of number of heads and number of outputs per head."
        assert src_len == value.size(0) and bsz == bsz_, "The number of elements in key and query should be equal"
        # Create the mask if attn_mask is None
        if attn_mask is None:
            attn_mask = torch.ones(tgt_len, src_len, dtype=torch.bool)
        # The number of heads dimension for query, key, and value
        hd_query = query.size(1)
        hd_key = key.size(1)
        hd_value = value.size(1)
        # Reshape query, key, and value (Batch size x Number of attention heads x Sequence length x Dimension of each attention head)
        query = query.view(tgt_len, bsz * hd_query, self.head_dim).transpose(0, 1) 
        key = key.view(src_len, bsz * hd_key, self.head_dim).transpose(0, 1)
        value = value.view(src_len, bsz * hd_value, self.head_dim).transpose(0, 1) 
        # Get the dot product of the query and the key (Batch size x Number of attention heads x Sequence length x Sequence length)
        attn_matmul_result = torch.matmul(query, key.transpose(-2, -1))
        attn_matmul_result = attn_matmul_result * self.scale
        attn_matmul_result.masked_fill_(attn_mask, float('-inf'))
        # Compute the attention weights
        attn_weight = torch.softmax(attn_matmul_result, dim=-1)
        attn_weight = self.attn_dropout(attn_weight)
        # Get the output after the attention
        attn_output = torch.matmul(attn_weight, value) 
        # Recover the shape of the output
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, hd_query * self.head_dim)
        # Get the final output
        output = self.proj_dropout(self.norm(attn_output))
        return output

class MultiHeadSelfAttention(MultiHeadAttention):
    def forward(self, hidden_states, attn_mask=None):
        # Project the hidden states
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        # Get the output of the attention
        output = super().forward(query, key, value, attn_mask=attn_mask)
        return output

# Initializing the model
m = MultiHeadSelfAttention(3, 2)

# Input tensor to the model
x = torch.randn(3, 4, 3)

attention_mask = torch.tensor([[0, 0, 1, 1],
                                [0, 0, 1, 1],
                                [1, 1, 1, 1],
                                [0, 0, 0, 1]], dtype=torch.bool)

output = m(x, attn_mask=attention_mask)
