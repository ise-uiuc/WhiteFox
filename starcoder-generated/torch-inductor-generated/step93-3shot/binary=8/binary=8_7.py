
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 9, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 10, 3, stride=2, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(x)
        v3 = self.conv3(x)
        return v3
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
