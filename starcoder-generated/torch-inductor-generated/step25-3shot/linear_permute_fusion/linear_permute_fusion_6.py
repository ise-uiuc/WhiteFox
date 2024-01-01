
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        v3 = x1 - v2
        v4 = x1 - 2
        v5 = torch.pow(v3, v4)
        return v5
# Inputs to the model
x1 = torch.randn(10, 2, 2)
