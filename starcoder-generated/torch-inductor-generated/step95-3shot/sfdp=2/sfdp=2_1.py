
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, emb, heads=8, dropout=0):
        super().__init__()

        self.emb = emb
        self.heads = heads
        self.dropout = dropout

        self.head_dim = emb
        if emb % heads!= 0:
          raise ValueError(f"Embedding dimension {emb} should be divisible by number of heads {heads}")
        self.depth = self.head_dim // self.heads

        self.query_w = nn.Linear(emb, emb, bias=False)
        self.key_w = nn.Linear(emb, emb, bias=False)
        self.value_w = nn.Linear(emb, emb, bias=False)
        self.layer_norm = nn.LayerNorm(emb)
    
    def forward(self, query, key, value, attn_mask=None):
        q = self.query_w(query)
        k = self.key_w(key)
        v = self.value_w(value)

        bs = q.size(0)

        qk = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.depth)
        qk.masked_fill_(attn_mask.unsqueeze(1) == 1, -1e4)

        attn = torch.softmax(qk, dim=-1)
        attn = torch.nn.functional.dropout(attn, p=self.dropout, training=self.training)

        output = torch.matmul(attn, v)
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(bs, -1, self.emb)

        result = self.layer_norm(output + query)

        return result

# Initializing the model
m = MultiHeadAttention(128)

# Input to the model
key = torch.randn(16, 128, 56, 56)
value = torch.randn(16, 128, 56, 56)
attn_mask = torch.ones(16, 128, 1, 56).cumsum(-1)!= 0
query = torch.randn(16, 128, 56, 56)
result = m(query, key, value, attn_mask)

