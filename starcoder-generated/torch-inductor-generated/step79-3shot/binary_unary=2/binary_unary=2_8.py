
class Model(torch.nn.Module):
    def __init__(self, v1, v2, v3, v4, v5, v6):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, v1, stride=v2, padding=v3)
        self.conv2 = torch.nn.Conv2d(8, 4, v4, stride=v5, padding=v6)
    def forward(self, x1):
        v2 = self.conv1(x1)
        v3 = v2 - 10
        v13 = F.relu(v3)
        v5 = self.conv2(v13)
        v6 = v5 - 11
        v15 = F.relu(v6)
        v7 = torch.squeeze(v15, 0)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
