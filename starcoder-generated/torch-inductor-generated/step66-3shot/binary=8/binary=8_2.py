
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3)
        self.linear2 = torch.nn.Linear(3, 2)
    def forward(self, x1):
        v1 = x1.flatten()
        v2 = v1.add(self.linear1.weight)
        v3 = v2 + self.linear1.bias
        v4 = v3.tanh()
        v5 = self.linear2(v1)
        v6 = torch.mul(v5, v4)
        v7 = v6.tanh()
        return v7
# Inputs to the model
x = torch.randn(1, 3)
