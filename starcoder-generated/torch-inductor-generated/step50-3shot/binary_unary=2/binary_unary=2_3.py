
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 4, stride=8, padding=2)
        self.conv2 = torch.nn.Conv2d(16, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 0.2
        v3 = F.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 - 0.8
        v6 = F.relu(v5)
        v7 = F.relu(v6 - 0.9)
        v8 = torch.squeeze(v7, 0)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
