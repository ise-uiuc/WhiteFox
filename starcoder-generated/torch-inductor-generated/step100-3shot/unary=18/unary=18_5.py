
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, (2, 3), 1, (0, 1))
        self.conv2 = torch.nn.Conv2d(1, 1, (1, 2), 1, padding=(1, 1))
    def forward(self, x2):
        v1 = (self.conv1(x2))
        v3 = torch.sigmoid(self.conv2(x2))
        return v3
# Inputs to the model
x2 = torch.randn(1, 1, 28, 28)
