
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1, 1)
        self.conv2 = torch.nn.Conv2d(1, 1, 1, 1)
        self.conv3 = torch.nn.Conv2d(1, 1, 3, 4, 2)
    def forward(self, x):
        x = self.conv1(x)
        x = torch.tanh(x)
        x = self.conv2(x)
        x = torch.tanh(x)
        x = self.conv3(x)
        x = torch.tanh(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 16, 16)
