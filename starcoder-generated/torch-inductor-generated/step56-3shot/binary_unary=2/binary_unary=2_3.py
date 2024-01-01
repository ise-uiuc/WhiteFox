
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(32, 8, 3, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 - 24
        v4 = self.conv3(v3.unsqueeze(0))
        v5 = v4 - 16
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 26, 26)
