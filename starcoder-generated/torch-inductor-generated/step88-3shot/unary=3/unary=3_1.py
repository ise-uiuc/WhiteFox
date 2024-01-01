
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1, 4)
        self.linear2 = torch.nn.Linear(4, 4)
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.linear2(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 1, 350)
