
class Model(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj = torch.nn.Linear(dim, dim)
 
    def forward(self, query, key, value):
        q = self.proj(query).view(query.size(0), query.size(1), 1, self.dim)
        k = self.proj(key).view(key.size(0), 1, -1, self.dim)
        res = torch.matmul(q, k.transpose(-2, -1))
        res = res / math.sqrt(self.dim)
        res = torch.nn.functional.softmax(res, dim=-1)
        res = torch.nn.functional.dropout(res, p=0.5)
        res = torch.matmul(res, value)
        return res

# Initializing the model
m = Model(512)

# Inputs to the model
query = torch.randn(1, 2, 512)
key = torch.randn(1, 2, 512)
value = torch.randn(1, 2, 512)
