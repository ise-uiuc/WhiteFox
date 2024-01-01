
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, kernel_size=1, stride=1)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.conv1.weight, self.conv1.bias)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 10, 10)
