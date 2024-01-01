
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.conv = torch.nn.Conv2d(1, 3, 1)
    def forward(self, x1):
        v1 = x1
        v2 = torch.nn.functional.linear(self.conv(v1), self.linear.weight, self.linear.bias)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 2, 2)
# Model begins
