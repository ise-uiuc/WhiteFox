
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 8, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(8, 4, 1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(4, 2, 1, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(2, 1, 1, stride=1, padding=1)
    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.tanh(self.conv3(x))
        x = torch.tanh(self.conv4(x))
        x = torch.tanh(self.conv5(x))
        return x
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
