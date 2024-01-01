
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 12)
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = 6 + v1
        v3 = 2 + v2
        v4 = 16.0 + 17.0 * v3
        return v4
# Inputs to the model
x1 = torch.randn(10, 10)
