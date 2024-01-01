
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(in_features=10, out_features=30)
        self.k = torch.nn.Linear(in_features=15, out_features=30)
 
    def forward(self, x1, x2):
        v1 = self.q(x1)
        v2 = self.k(x2)
        v3 = torch.matmul(v1, v2.transpose(-2, -1))
        v4 = v3 / 30.0
        v5 = v4.softmax(dim=-1)
        v6 = torch.matmul(v5, v2)
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 20, 10)
x2 = torch.randn(2, 30, 15)
