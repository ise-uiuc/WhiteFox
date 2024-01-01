
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 8, 1, 1, 0)
        self.conv2 = torch.nn.Conv2d(8, 8, 3, 1, 1)
        self.conv3 = torch.nn.Conv2d(8, 16, 3, 1, 1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(torch.relu(v1))
        v3 = self.conv3(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 16, 640,640)
