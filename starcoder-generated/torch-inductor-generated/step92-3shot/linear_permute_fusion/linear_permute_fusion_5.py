
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.linear2 = torch.nn.Linear(1, 1)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        v3 = torch.nn.functional.linear(v2, self.linear2.weight, self.linear2.bias)
        v4 = v3.permute(0, 2, 1)
        return v1 + v4 + x1
# Inputs to the model
x1 = torch.randn(1, 1, 1)
