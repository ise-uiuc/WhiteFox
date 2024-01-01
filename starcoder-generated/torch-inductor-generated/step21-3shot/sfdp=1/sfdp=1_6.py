
class Model(torch.nn.Module):
    def __init__(self, dim, num_heads, dropout_p):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.scaled_dot_product = ScaledDotProductAttention(dim, dropout_p, scale=1. / num_heads)
 
    def forward(self, query, key, value):
        v1_t = query.transpose(-2, -1)
        v2 = torch.matmul(key, v1_t)
        v3 = v2.div(math.sqrt(self.dim))
        v4 = self.scaled_dot_product(v3, v3, v2)
        return v4

# Initializing the model
m = Model(128, 4, 0.5)

# Inputs to the model
query = torch.randn(4, 4, 128)
key = torch.randn(4, 6, 128)
value = torch.randn(4, 6, 128)
