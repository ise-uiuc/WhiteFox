
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 1, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(1, 1, 1, stride=1, padding=1)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = torch.tanh(x1)
        x3 = self.conv2(x2)
        x4 = torch.tanh(x3)
        x5 = self.conv3(x4)
        x6 = torch.tanh(x5)
        return x6
# Inputs to the model
x = torch.rand(1, 1, 49, 89)
