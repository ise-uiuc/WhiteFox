
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 1, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 3, 1, stride=2, padding=1)
    def forward(self, x1):
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x4 = x3 - 0.15
        x5 = F.relu(x4)
        return x5
# Inputs to the model
x1 = torch.randn(1, 3, 512, 512)
