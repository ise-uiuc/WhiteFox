
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 19, 3, stride=2, padding=9)
        self.conv2 = torch.nn.Conv2d(19, 29, 3, stride=1, padding=1)
    def forward(self, x1):
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x4 = x3 - 0.1
        x5 = F.relu(x4)
        return x5
# Inputs to the model
x1 = torch.randn(1, 16, 20, 20)
