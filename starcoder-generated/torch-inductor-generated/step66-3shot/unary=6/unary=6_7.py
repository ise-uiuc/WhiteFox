
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 100, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(100, 3, 3, stride=1, padding=1)
    def forward(self, x1):
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x1 = torch.sigmoid(x1)
        return x1
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
