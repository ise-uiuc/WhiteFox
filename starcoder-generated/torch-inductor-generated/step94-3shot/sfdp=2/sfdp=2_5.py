
class MultiHeadAttention(Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * self.num_heads == self.embed_dim, "embedding dimension = num_heads * head_dim"
 
        self.k_proj = Linear(self.embed_dim, embed_dim)
        self.q_proj = Linear(self.embed_dim, embed_dim)
        self.v_proj = Linear(self.embed_dim, embed_dim)
        self.dropout = Dropout(attn_pdrop)
 
    def forward(self, query, key, value):
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
 
        q = q.view(query.size(0), query.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(-1, key.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(-1, value.size(1), self.num_heads, self.head_dim).transpose(1, 2)
 
        attn_weights = torch.matmul(q, k.transpose(2, 3))
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)
 
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(attn_out.size(0), -1, self.embed_dim)
        return attn_out
        
# Initializing the model
m = MultiHeadAttention(embed_dim, num_heads)

# Inputs to the model
query = torch.randn(query_len, bsz, embed_dim)
key = torch.randn(key_len, bsz, embed_dim)
value = torch.randn(key_len, bsz, embed_dim)
# In the forward method, please use the output of each of the three layers after initialization.
# The result is stored in the variable __output__.
