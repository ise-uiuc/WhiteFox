
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 4, 1, stride=1, padding=1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.sigmoid(x)
        x = x * x
        x = self.conv2(x)
        x = self.sigmoid(x)
        x = x * x
        return x
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
