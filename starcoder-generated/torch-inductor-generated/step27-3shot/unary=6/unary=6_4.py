
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 1, 5, stride=1, padding=2)
        self.conv3 = torch.nn.Conv2d(1, 1, 7, stride=1, padding=3)
    def forward(self, x1):
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        return torch.mean(torch.cat([x2, x3, x4], dim=1), dim=1, keepdim=True)
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
