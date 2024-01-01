
class Model(torch.nn.Module):
    def __init__(self, key_dim, query_dim, value_dim):
        super().__init__()
        self.key_dim = key_dim
        self.query_dim = query_dim
        self.value_dim = value_dim
 
        self.wq = torch.nn.Linear(query_dim, query_dim, bias=False)
        self.wk = torch.nn.Linear(key_dim, key_dim, bias=False)
        self.wv = torch.nn.Linear(value_dim, value_dim)
        self.dropout = torch.nn.Dropout2d(p=0.0)
 
    def forward(self, x1, x2, x3):
        v1 = self.wq(x1)
        v2 = self.wk(x2)
        v3 = torch.matmul(v1, v2.transpose(-2, -1)) / math.sqrt(v1.shape[-1])
        v4 = v3.softmax(dim=-1)
        v5 = self.dropout(x3)
        v6 = self.wv(v5)
        v7 = torch.matmul(v4, v6)
        return v7

# Initializing the model
m = Model(64, 512, 512)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 512)
x3 = torch.randn(1, 3, 512)
