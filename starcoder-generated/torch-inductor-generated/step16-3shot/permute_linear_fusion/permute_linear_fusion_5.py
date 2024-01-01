
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = self.linear.weight * v1
        x2 = self.linear(v2)
        x1 = v1 * x2
        x3 = torch.sigmoid(x1)
        v1 = x1 * x3
        x3 = x3 + x1
        x3 = v1 + x3
        x2 = x2 * x3
        return x2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
