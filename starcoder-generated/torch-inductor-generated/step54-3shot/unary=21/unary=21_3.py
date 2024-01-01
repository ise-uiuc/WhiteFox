
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 3, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 28, 28)
