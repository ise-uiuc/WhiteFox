
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.conv = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        v1 = v2.permute(0, 1, 3, 2)
        v2 = self.conv(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
