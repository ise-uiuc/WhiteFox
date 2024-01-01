
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
    def forward(self, x1, x2, x3):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v3 = torch.nn.functional.linear(x2, self.linear.weight, self.linear.bias)
        v2 = v3.permute(0, 2, 1)
        return v1 + v2 + v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(3, 2, 2)
x3 = torch.randn(1, 2, 2)
