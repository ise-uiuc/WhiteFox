
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x, y):
        x1 = x.permute(0, 2, 1)
        x2 = y.permute(0, 2, 1)
        x3 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        x4 = torch.nn.functional.linear(x2, self.linear.weight, self.linear.bias)
        return x3 + x4
# Inputs to the model
x = torch.randn(1, 2, 2)
y = torch.randn(1, 2, 2)
