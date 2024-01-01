
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 30, 1)
        self.conv2 = torch.nn.Conv2d(30, 30, 1)
        self.conv3 = torch.nn.Conv2d(30, 40, 1)
    def forward(self, input):
        x = self.conv1(input)
        x = torch.tanh(x)
        x = self.conv2(x)
        x = torch.tanh(x)
        x = self.conv3(x)
        x = torch.tanh(x)
        return x
# Inputs to the model
input = torch.randn(1, 3, 32, 32)
