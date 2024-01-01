
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 30, 1)
        self.conv2 = torch.nn.Conv2d(30, 30, 1)
        self.conv3 = torch.nn.Conv2d(30, 30, 1)
        self.conv4 = torch.nn.Conv2d(30, 40, 1)
    def forward(self, input):
        x = self.conv1(input)
        y = self.conv2(x)
        y = torch.tanh(y)
        z = self.conv3(y)
        s = self.conv4(z)
        return torch.tanh(s)
# Inputs to the model
input = torch.randn(1, 3, 32, 32)
