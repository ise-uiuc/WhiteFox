
class MultiHeadAttentionLayer(torch.nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.2, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
 
        self.num_heads = num_heads
        self.dropout = dropout
 
        assert embed_dim % num_heads == 0, f"Embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
 
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
 
        self.wq = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.wk = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.wv = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
 
        self.dense = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
 
    def forward(self, inputs):
        v = self.wq(inputs)
        k = self.wk(inputs)
        q = self.wq(inputs)
 
        v = v.chunk(self.num_heads, dim=-1)
        k = k.chunk(self.num_heads, dim=-1)
        q = q.chunk(self.num_heads, dim=-1)
 
        return v, k, q
 
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mhalayer = MultiHeadAttentionLayer(128)
 
    def forward(self, x1, x2, x3):
        v, k, q = self.mhalayer(x1)
        v = torch.stack(v)
        k = torch.stack(k)
        q = torch.stack(q)
 
        value_len = v.size()[-2]
        key_len = k.size()[-2]
        query_len = q.size()[-2]
        causal_mask = torch.triu(torch.ones(query_len, key_len), 1+query_len-key_len).bool().to(x1.device)
 
        attn_mask = torch.tril(torch.ones(query_len, value_len, device=x1.device)).bool()
        attn_mask += causal_mask
 
        return v, k, q, attn_mask
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(24, 128, 8)
x2 = torch.randn(24, 128, 8)
x3 = torch.randn(24, 128, 8)
__output__, __tmp__, __tmp__, __tmp__ = m(x1, x2, x3)


