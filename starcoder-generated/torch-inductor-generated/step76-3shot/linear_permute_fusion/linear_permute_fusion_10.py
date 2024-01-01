
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x0):
        v0 = x0
        v1 = self.pool(self.linear(v0))
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = self.pool(v2)
        v4 = v3 + v2
        return v4
# Inputs to the model
x0 = torch.randn(1, 2, 2)
