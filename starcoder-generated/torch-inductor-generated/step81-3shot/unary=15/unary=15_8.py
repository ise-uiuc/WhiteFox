
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(160, 640, 5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(640, 1024, 5, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = torch.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 160, 1024, 1024)
