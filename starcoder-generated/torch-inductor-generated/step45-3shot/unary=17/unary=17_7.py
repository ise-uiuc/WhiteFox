
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, 3, padding=1, stride=1)
        self.conv2 = torch.nn.ConvTranspose2d(10, 10, 3, padding=2, stride=2, bias=False)
        self.conv3 = torch.nn.Conv2d(10, 1, 5, padding=2, stride=1)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        return x
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
