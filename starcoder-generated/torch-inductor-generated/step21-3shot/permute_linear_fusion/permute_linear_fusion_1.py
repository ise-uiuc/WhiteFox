
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)
    def forward(self, x1):
        v1 = x1 + self.linear.weight
        x1 = self.linear(v1)
        v2 = torch.abs(x1)
        return v1 * v2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
