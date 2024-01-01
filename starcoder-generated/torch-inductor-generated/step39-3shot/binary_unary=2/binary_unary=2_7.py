
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 8, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.mean(v1, dim=[-1, -2])
        v3 = torch.squeeze(v2, dim=-1)
        v4 = v3 - 5.6
        v5 = F.relu(v4)
        v6 = self.conv2(v5)
        v7 = torch.sum(v6, dim=[-1, -2])
        v8 = torch.squeeze(v7, dim=-1)
        v9 = torch.mean(v8, dim=[-1, -2])
        v10 = v9 - 7
        v11 = F.relu(v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
