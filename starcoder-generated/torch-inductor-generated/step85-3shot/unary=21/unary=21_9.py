
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 128, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(128, 3, 1, stride=1, padding=1)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.conv3(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
