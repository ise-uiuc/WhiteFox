
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 2, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(2, 2, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = v1.add_(v2)
        # v4 = torch.relu(v3)
        v4 = v3.add_(2)
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 64, 64)
