
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2,2)
    def forward(self, x1):
        v1 = self.linear.bias
        v2 = self.linear.weight
        v3 = v1 + v2
        x2 = v1.permute(0, 2, 1)
        v3 = v3 + x2
        v3 = v3.permute(0, 2, 1)
        x3 = x1 + v3
        v4 = torch.rand(2, 2)
        x4 = x1 * v2
        x3 = x4 - x1
        x2 = x3 + 1
        return x2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
