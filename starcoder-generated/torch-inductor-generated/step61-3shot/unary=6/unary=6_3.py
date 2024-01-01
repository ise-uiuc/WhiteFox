
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 11, 3, stride=3, padding=4)
        self.bn = torch.nn.BatchNorm2d(3, eps=1e-05, momentum=0.1)
        self.hardtanh = torch.nn.Hardtanh(inplace=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = F.relu(v1)
        v3 = v2.permute(0, 2, 3, 1).unsqueeze(-1)
        return self.bn(v3)
# Inputs to the model
x1 = torch.randn(1, 3, 192, 192)
