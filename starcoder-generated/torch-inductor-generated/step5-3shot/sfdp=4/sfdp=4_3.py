
class SelfAttention(torch.nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = torch.nn.Linear(embed_dim, embed_dim)
        self.key = torch.nn.Linear(embed_dim, embed_dim)
        self.value = torch.nn.Linear(embed_dim, embed_dim)
 
    def forward(self, query, key, value, attn_mask=None):
        q = self.query(query)
        k = self.key(key)
        v = self.value(value)
 
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(self.embed_dim)
        if attn_mask is not None:
            scores = scores + attn_mask
 
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output

# Initializing the model
m = SelfAttention(embed_dim=128)

# Inputs to the model
x1 = torch.randn(1, 2, 128)
x2 = torch.randn(1, 2, 128)
x3 = torch.randn(1, 2, 128)
attn_mask = torch.tensor([[1, 0], [0, 0]])
