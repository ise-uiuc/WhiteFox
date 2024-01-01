
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,1, 28, 1)
        self.tanh = torch.nn.Tanh()
        self.conv2 = nn.Conv2d(1, 1, 1, 1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.conv2(x)
        x = self.tanh(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 32, 32)
