
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(3, 64, 1, stride=1)
        self.conv_2 = torch.nn.Conv2d(64, 256, 4, stride=2, padding=1)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = self.relu(v1)
        v3 = self.conv_2(v2)
        v4 = self.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)
