
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        v4 = x1 - v2
        v3 = torch.nn.functional.linear(v4, self.linear.weight, self.linear.bias)
        return x1 - v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
