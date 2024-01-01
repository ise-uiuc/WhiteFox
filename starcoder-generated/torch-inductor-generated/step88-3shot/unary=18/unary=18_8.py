
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=1)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    def forward(self, x1):
        v1 = torch.relu(self.conv1(x1))
        v2 = torch.relu(self.maxpool1(v1))
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
