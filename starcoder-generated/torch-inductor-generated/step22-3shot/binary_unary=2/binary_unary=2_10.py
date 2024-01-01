
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 8, stride=2, padding=2)
        self.conv2 = torch.nn.Conv2d(8, 16, 4, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 0.5
        v3 = F.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 - 0.5
        v6 = F.relu(v5)
        v7 = torch.squeeze(v6, 0)
        return v7
# Inputs to the model
x1 = torch.randn(1, 1, 128, 128)
