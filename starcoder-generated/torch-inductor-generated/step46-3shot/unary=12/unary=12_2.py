
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=2, padding=1, dilation=2)
    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        return x
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
