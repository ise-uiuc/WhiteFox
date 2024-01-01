
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, 2, stride=2, padding=1)
        self.flatten = torch.flatten
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.flatten(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
