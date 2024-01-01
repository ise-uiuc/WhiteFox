
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 32, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 2, 1, stride=1, padding=1)
    def forward(self, x1):
        x1 = F.relu(self.conv2(F.relu(self.conv(x1))))
        return x1
# Inputs to the model
x1 = torch.randn(1, 64, 64, 64)

