
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 1)
        self.conv2 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 16, 1, padding=1)
    def forward(self, x1):
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        return x1
# Inputs to the model
x1 = torch.randn(1, 16, 4, 4)
