
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 3)
        self.relu = torch.nn.ReLU()
        self.pool1 = torch.nn.AdaptiveMaxPool2d(3)
        self.conv1 = torch.nn.Conv2d(8, 16, 1)
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv1(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 8, 8)
