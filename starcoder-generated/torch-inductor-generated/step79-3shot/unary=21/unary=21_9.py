
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 3, stride=2)
        self.conv2 = torch.nn.Conv2d(16, 128, 3, stride=2)
        self.conv3 = torch.nn.Conv2d(128, 512, 3, stride=2)
    def forward(self, x):
        x = self.conv1(x)
        x = torch.tanh(x)
        x = self.conv2(x)
        x = torch.tanh(x)
        x = self.conv3(x)
        x = torch.tanh(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 56, 56)
