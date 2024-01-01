
class Model(torch.nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.q = torch.nn.Linear(dim, dim)
        self.k = torch.nn.Linear(dim, dim)
        self.v = torch.nn.Linear(dim, dim)
        self.ff = torch.nn.Sequential(torch.nn.Linear(dim, dim),
                                      torch.nn.GELU(),
                                      torch.nn.Linear(dim, dim))
        self.ln = torch.nn.LayerNorm(dim, eps=1e-6)
        self.drop = torch.nn.Dropout(dropout_p)
 
    def forward(self, query, key, value, mask=None):
        qkv = (query + self.q(key) + self.k(value)) / 3
        qkv = self.ln(qkv)
        out = self.drop(self.ff(qkv))
        return out

# Initializing the model
m = Model(dim, num_heads)
m.train()

# Inputs to the model
query = torch.randn(batch_size, num_queries, size_hidden)
key = torch.randn(batch_size, num_key_value, size_hidden)
value = torch.randn(batch_size, num_key_value, size_hidden)
