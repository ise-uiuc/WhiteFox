
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(6, 6)
    def forward(self, x0):
        v0 = torch.nn.functional.linear(x0, self.linear.weight, self.linear.bias)
        v1 = v0.permute(2, 0, 1)
        v2 = v1.permute(0, 2, 1)
        v3 = v1.permute(2, 0, 1)
        v4 = v1.permute(1, 2, 0)
        return v0 + v2 + v3 + v4
# Inputs to the model
x0 = torch.randn(1, 2, 3)
