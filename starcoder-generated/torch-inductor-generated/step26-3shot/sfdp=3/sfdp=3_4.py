
class Model(torch.nn.Module):
    def __init__(self, m, n, p, q):
        super().__init__()
        self.m = m
        self.n = n
        self.p = p
        self.q = q
        self.dropout = torch.nn.Dropout(p)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.matmul5 = torch.matmul(self.m, self.n.transpose(-2, -1)) # Use this matmul in the forward() function
 
    def forward(self, x1):
        _scale_factor1 = torch.tensor(1 / 64.0)
        _scale_factor = _scale_factor1.to(torch.float32)
        v1 = torch.matmul(x1, self.matmul5)
        v2 = v1.mul(_scale_factor)
        v3 = self.softmax(v2)
        v4 = self.dropout(v3)
        v5 = torch.matmul(v4, self.q)
        return v5

# Initializing the model
m = Model()
n = torch.randn(512, 512)
p = 0.25
q = torch.randn(24, 512)
x1 = torch.randn(24, 512, 64)
