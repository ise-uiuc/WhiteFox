
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(64, 1, 5, stride=1, padding=2)
    def forward(self, x3):
        v1 = self.conv1(x3)
        v2 = self.relu1(v1)
        v3 = self.conv2(v2)
        return v3
# Inputs to the model
x3 = torch.randn(1, 32, 128, 32)
