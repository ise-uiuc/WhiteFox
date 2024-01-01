
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 96, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv2 = torch.nn.Conv2d(192, 64, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = torch.relu(v2)
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(2, 64, 96, 96)
