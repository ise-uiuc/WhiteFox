
class Model(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.scale = dim ** -0.5
        self.to_keys = nn.Linear(dim, dim * num_heads, bias=False)
        self.to_values = nn.Linear(dim, dim * num_heads, bias=False)
        self.to_out = nn.Linear(dim * num_heads, dim)

    def forward(self, x):
        b, t, d, h = *x.shape, self.num_heads
        keys    = self.to_keys (x).view(b, t, d, h)
        queries = self.to_keys (x).view(b, t, d, h)
        values  = self.to_values(x).view(b, t, d, h)
        scaled_dot_product = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        attention_weights = scaled_dot_product.softmax(dim=-1)
        out = torch.matmul(attention_weights, values)
        return self.to_out(out)

# Initializing the model
m = Model(dim=512)

# Inputs to the model
x = torch.randn(1, 100, 512)
