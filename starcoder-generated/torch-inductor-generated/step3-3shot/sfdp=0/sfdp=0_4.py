
class Model(torch.nn.Module):
    def __init__(self, key_dim, num_heads):
        super().__init__()

        self.key_dim = key_dim
        self.num_heads = num_heads
        self.q_proj = torch.nn.Linear(key_dim, num_heads * key_dim)
        self.k_proj = torch.nn.Linear(key_dim, num_heads * key_dim)
        self.v_proj = torch.nn.Linear(key_dim, num_heads * key_dim)
        self.fc = torch.nn.Linear(num_heads * key_dim, key_dim)

    def forward(self, x1, x2):
        q = self.q_proj(x1)
        k = self.k_proj(x2)
        v = self.v_proj(x2)
        # __init__ doesn't provide arguments for query and key
        d_k = self.key_dim
        scale = 1 / math.sqrt(d_k)
        q = q * scale
        x = torch.matmul(q, k.transpose(-2, -1))
        x = x.softmax(dim=-1)
        return self.fc(x.matmul(v))

# Inputs to the model
key = torch.randn(1, 1, 3, 3)
query = torch.randn(1, 2, 3, 3)
