
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 32, 3, stride=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, stride=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(128, 16, 3, stride=1, padding=1)
    def forward(self, x):
        x = self.conv1(x)
        x = torch.tanh(x)
        x = self.conv2(x)
        x = torch.tanh(x)
        x = self.conv3(x)
        x = torch.tanh(x)
        x = self.conv4(x)
        return x
# Inputs to the model
x = torch.randn(1, 16, 256, 256)
