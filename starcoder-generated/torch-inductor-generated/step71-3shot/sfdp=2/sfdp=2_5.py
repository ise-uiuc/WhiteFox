
class Model(torch.nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.1)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mat1 = torch.nn.Parameter(torch.randn(n, d) * 0.01, requires_grad=True)
        self.mat2 = torch.nn.Parameter(torch.randn(n, d) * 0.01, requires_grad=True)
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, self.mat1)
        v2 = torch.matmul(x2, self.mat2)
        v3 = v1 + v2
        v4 = self.dropout(v3)
        v5 = self.softmax(v4)
        v6 = v5 * v3
        return v6

# Initializing the model
n = 7
d = 4
m = Model(n, d)

# Inputs to the model
x1 = torch.randn(2, n, d)
x2 = torch.randn(2, n, d)
