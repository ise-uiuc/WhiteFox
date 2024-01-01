
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(6, 128, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(128, 128, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.relu(v3)
        v5 = v2 + v4
        return v5
# Inputs to the model
x1 = (torch.randn(1, 6, 512, 512), torch.randn(1, 6, 512, 512))
