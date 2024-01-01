
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(64, 10)
        self.l2 = torch.nn.Linear(10, 5)
 
    def forward(self, x1, x2, x3):
        v1 = self.l1(x1)
        v2a = self.l2(v1)
        v2 = torch.matmul(v2a, x2)
        v3 = torch.matmul(v2a, x3)
        v3 = torch.transpose(v3, -2, -1)
        v4 = torch.softmax(v3, dim=-1)
        v5 = torch.matmul(v4, v2)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64)
x2 = torch.rand(1, 4, 10)
x3 = torch.rand(1, 5, 10)
