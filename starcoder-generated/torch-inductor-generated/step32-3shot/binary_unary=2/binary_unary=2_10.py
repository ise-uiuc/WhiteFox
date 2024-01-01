
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = (v1 - 1)
        v3 = self.conv1(v2)
        t1 = (v3 - 2)
        v4 = self.conv2(t1)
        t2 = (v4 - 3)
        return F.relu(t2)
# Inputs to the model
x1 = torch.randn(1, 3, 100, 100)
