
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.1597118722272973)
        self.softmax = torch.nn.Softmax(dim=-1)
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v1 = v1 / math.sqrt(x1.size(-1))
        v1 = v1 + (-1e7 * (x1 == -1))
        v2 = self.softmax(v1)
        v3 = self.dropout(v2)
        v4 = torch.matmul(v3, x2)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10, 4)
x2 = torch.randn(1, 12, 6)
