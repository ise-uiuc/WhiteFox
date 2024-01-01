
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 32, 3, dilation=2, padding=2)
        self.conv2 = torch.nn.Conv2d(32, 16, 1, dilation=2)
        self.conv3 = torch.nn.Conv2d(16, 8, 3, dilation=1)
        self.conv4 = torch.nn.Conv2d(8, 8, 1, dilation=5)
    def forward(self, input):
        x = self.conv1(input)
        x = torch.tanh(x)
        x = self.conv2(x)
        x = torch.tanh(x)
        x = self.conv3(x)
        x = torch.tanh(x)
        x = self.conv4(x)
        x = torch.tanh(x)
        return x
# Inputs to the model
input = torch.randn(1, 16, 10, 10)
