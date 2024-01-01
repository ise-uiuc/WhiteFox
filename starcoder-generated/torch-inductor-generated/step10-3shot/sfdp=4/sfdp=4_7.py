
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 4
        self.num_heads = 2
        self.head_dim = 1
        self.multihead_attn = torch.nn.MultiheadAttention(self.embed_dim, self.num_heads, dropout=0., bias=False, add_bias_kv=False, add_zero_attn=False)
 
    def forward(self, query, key, value, attn_mask=None):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ value
        return output
 
# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 2, 5, m.embed_dim)
key = torch.randn(1, 2, 4, m.embed_dim)
value = torch.randn(1, 2, 4, m.embed_dim)
attn_mask = -torch.eye(4).unsqueeze(dim=0).unsqueeze(dim=0)
