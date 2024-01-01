
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 16, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=3)
    def forward(self, x1):
        x = self.conv1(x1)
        x = self.conv2(x)
        return x
# Inputs to the model
x1 = torch.randn(1, 4, 16, 16)
