
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
        self.linear2 = torch.nn.Linear(5, 5)
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + 3
        v3 = v2.clamp(min=0, max=6)
        v4 = v1 * v3
        v5 = v4.div(6)
        v6 = self.linear2(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 10)
