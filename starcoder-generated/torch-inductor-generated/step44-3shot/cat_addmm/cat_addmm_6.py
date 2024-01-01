
class Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 3)
        self.relu1 = nn.ReLU(1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = nn.MaxPool2d(x, dim=2)
        return x
# Inputs to the model
x = torch.randn(2, 1, 3, 3)
