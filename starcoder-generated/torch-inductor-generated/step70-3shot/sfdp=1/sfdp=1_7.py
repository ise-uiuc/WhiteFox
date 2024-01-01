
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, dim_V):
        super(MultiHeadAttention, self).__init__()
        d_k = dim_V  # the dimension of the key, the dimension also equals that of the output
        num_head = 10  # we implement multi-head attention using 10 parallel attention blocks
        self.W = torch.nn.Linear(dim_V, num_head * d_k, bias=False)
        self.E = torch.nn.Linear(d_k, num_head * d_k, bias=False)
        self.v = torch.nn.Linear(dim_V, num_head * 1, bias=False)
 
    def forward(self, query, key, value):
        q = self.W(query)
        k = self.E(key).transpose(-2, -1)
        V = self.v(value)
        x1 = torch.matmul(q, k)
        v1 = torch.tensor(q.size(-1), dtype=torch.float32).to(output.device)
        return torch.matmul(v1.div(np.sqrt(k.size(-1))), x1.transpose(-2, -1)).matmul(V)

# Initializing the model
m = MultiHeadAttention(768)

# Inputs to the model
query = torch.randn(10, 1, 768, 75)
key = torch.randn(10, 1, 768, 157)
value = torch.randn(10, 1, 768, 157)
