
class MultiheadAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.w_k = torch.nn.Linear(embed_dim, embed_dim)
        self.w_q = torch.nn.Linear(embed_dim, embed_dim)
        self.w_v = torch.nn.Linear(embed_dim, embed_dim)
        self.dot_product_attention = DotProductAttention(dropout)
        self.linear = torch.nn.Linear(embed_dim, embed_dim)
 
    def forward(self, q, k, v):
        k = self.w_k(k)
        q = self.w_q(q)
        v = self.w_v(v)
        q, k, v = split_heads_2(q, k, v, self.num_heads)
        attention = self.dot_product_attention(q, k, v)
        attention = combine_heads_2(attention)
        output = self.linear(attention)
        return output

# Initializing the model
m = MultiheadAttention(embed_dim=128, num_heads=16)

# Inputs to the model
batch_size, max_src_length, max_tgt_length = 1, 100, 100
h = torch.randn(batch_size, max_src_length, max_tgt_length, 128)
