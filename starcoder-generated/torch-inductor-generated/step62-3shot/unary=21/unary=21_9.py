
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, stride=2)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, stride=2)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, stride=2)
        self.conv4 = torch.nn.Conv2d(64, 128, 3, stride=2)
    def forward(self, x):
        x = self.conv1(x)
        x = torch.tanh(x)
        x = self.conv2(x)
        x = torch.tanh(x)
        x = self.conv3(x)
        x = torch.tanh(x)
        x = self.conv4(x)
        return torch.tanh(x)

# Inputs to the model
x = torch.randn(1, 3, 128, 128)
