
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(20, 14, 3, stride=1, padding=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(1, affine=False, track_running_stats=True)
        self.relu = torch.nn.ReLU6()  # Relu6 is clamping to a range [0, 6]
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, x1):
        x1 = self.conv(x1)
        x1 = self.bn(x1)
        x1 = self.relu(x1)
        x1 = self.softmax(x1)
        return x1

# Inputs to the model
x1 = torch.randn(3, 20, 200, 200)
