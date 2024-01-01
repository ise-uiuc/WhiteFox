
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qk = torch.nn.Linear(6, 4)
 
    def forward(self, x1, x2, x3):
        v1 = self.qk(x1)
        v2 = torch.matmul(v1, x2.transpose(-2, -1))
        v3 = v2 / 16.0
        v4 = F.softmax(v3, dim=-1)
        v5 = F.dropout(v4, p=0.3)
        v6 = torch.matmul(v5, x3)
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 6, 32, 4)
x2 = torch.randn(1, 32, 4, 8)
x3 = torch.randn(1, 4, 32, 8)
