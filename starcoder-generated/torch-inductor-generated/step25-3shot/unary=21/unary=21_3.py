
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.conv2 = torch.nn.Conv2d(64, 32, 1, stride=1, padding=0, bias=False)
        self.conv3 = torch.nn.Conv2d(32, 64, 1, stride=1, padding=0, bias=False)
        self.conv4 = torch.nn.Conv2d(64, 32, 3, stride=1, padding=1, bias=False)
    def forward(self, x):
        x = self.conv1(x)
        x = torch.tanh(x)
        x = self.conv2(x)
        x = torch.sigmoid(x)
        x = self.conv3(x)
        x = torch.tanh(x)
        x = self.conv4(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
