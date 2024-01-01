
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 5, stride=2, padding=0)
        self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.relu(v1)
        v3 = self.conv(v2)
        v4 = torch.relu(v3)
        v6 = self.pool(v4)
        return v6
# Inputs to the model
x = torch.randn(1, 3, 100, 100)
