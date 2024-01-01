
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 4, 1)
        self.conv2 = torch.nn.Conv2d(4, 16, 3)
        self.conv3 = torch.nn.Conv2d(16, 8, 1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        t1 = v3 * 1.632
        v4 = torch.relu(t1)
        return v4
# Inputs to the model
x1 = torch.randn(1, 8, 32, 32)
