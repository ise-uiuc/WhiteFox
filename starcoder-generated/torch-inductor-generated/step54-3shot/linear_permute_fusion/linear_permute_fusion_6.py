
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.conv2d = torch.nn.Conv2d(1, 1, 3)
    def forward(self, x):
        v1 = torch.nn.functional.linear(x, self.linear.weight, self.linear.bias)
        return self.conv2d(v1)
# Inputs to the model
x = torch.randn(1, 2, 2)
