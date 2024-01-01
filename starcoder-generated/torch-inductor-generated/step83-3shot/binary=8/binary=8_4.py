
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = v1 + v2
        v3 = torch.transpose(v3, 1, 2)
        v4 = v3.view(v3.shape[0], -1)
        v5 = v4.sum(axis=1).unsqueeze(-1).unsqueeze(-1)
        v6 = v5.repeat(1, 1, v3.shape[2], v3.shape[3])
        v7 = v3 + v6
        v8 = self.bn1(v7)
        v9 = v3.mean(axis=-2).unsqueeze(-2).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, v3.shape[2], v3.shape[3])
        v10 = v7 + v8 + v9
        return v10
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
