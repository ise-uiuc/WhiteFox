
class Model(torch.nn.Module):
    def __init__(self, dim=2, dim_head=8, heads=4, dropout=0):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
 
        self.to_queries = torch.nn.Linear(dim, heads * dim_head)
        self.to_keys    = torch.nn.Linear(dim, heads * dim_head)
        self.to_values  = torch.nn.Linear(dim, heads * dim_head)
        self.after_norm = torch.nn.Linear(heads * dim_head, dim)
        self.dropout    = torch.nn.Dropout(dropout)
 
    def forward(self, queries, keys, values):
        b, h, device = *queries.shape[:2], queries.device
        h = self.heads
        q, k, v = (self.to_queries(queries), self.to_keys(keys), self.to_values(values))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = dots.softmax(dim=-1)
        dropout_attn = self.dropout(attn)
        out = torch.matmul(dropout_attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.after_norm(out)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16, 1024)
x2 = torch.randn(1, 16, 1024)
x3 = torch.randn(1, 16, 1024)
