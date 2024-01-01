
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3)
        self.conv2 = torch.nn.Conv2d(32, 64, 2)
        self.conv3 = torch.nn.Conv2d(64, 128, 1)
        self.conv4 = torch.nn.Conv2d(128, 32768, 6)
        self.conv5 = torch.nn.ConvTranspose2d(32768, 3, 1)
    def forward(self, x):
        x = self.conv1(x)
        x = torch.tanh(x)
        x = self.conv2(x)
        x = torch.tanh(x)
        x = self.conv3(x)
        x = torch.tanh(x)
        x = self.conv4(x)
        x = torch.tanh(x)
        x = self.conv5(x)
        x = torch.tanh(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 127, 127)
