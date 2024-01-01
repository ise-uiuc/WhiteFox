
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, bias=False)
        self.activation = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm2d(3)
        self.conv2 = torch.nn.Conv2d(3, 3, 3, bias=False)
        self.conv2d = torch.nn.Conv2d(3, 3, 3, bias=False)
    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.bn(x)
        x = self.activation(self.conv2(x))
        x = x + self.conv2d(F.relu(x))
        return x
# Inputs to the model
x = torch.randn(1, 3, 6, 6)
