
class Model(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, num_query, num_key):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_query = num_query
        self.num_key = num_key
        self.inv_scale = torch.rsqrt(torch.tensor(self.embed_dim // self.num_heads, dtype=torch.float32))
        self.q = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.k = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.v = torch.nn.Linear(self.embed_dim, self.embed_dim)
 
    def forward(self, x1):
        q = self.q(x1[:, 0, :])
        k = self.k(x1[:, 1, :])
        v = self.v(x1[:, 2, :])
        scaled_dot_product = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attention_weights = scaled_dot_product.softmax(dim=-1)
        output = attention_weights.matmul(v)
        return output
 
 
# Initializing the model
embed_dim = 512
num_heads = 8
num_query = 2
num_key = 1
x1 = torch.randn(8, 3, 512)
m = Model(embed_dim, num_heads, num_query, num_key)
