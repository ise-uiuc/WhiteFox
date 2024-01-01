
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(6)
        self.conv2 = torch.nn.Conv2d(6, 8, 3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(8)
    def forward(self, x1):
        v0 = self.bn1(self.conv1(x1))
        v0 = F.relu(v0)
        v0 = F.relu(self.bn2(self.conv2(v0)))
        v0 = v0.view(v0.size(0), -1)
        v0 = torch.sum(v0, dim=1, keepdim=True)
        return v0
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
