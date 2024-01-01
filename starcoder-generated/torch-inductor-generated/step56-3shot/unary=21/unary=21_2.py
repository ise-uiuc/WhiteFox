
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 2, 2, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(2, 2, 2, stride=2)
        self.conv3 = torch.nn.Conv2d(2, 1, 2)
    def forward(self, input):
        x = self.conv1(input)
        x = torch.tanh(x)
        x = self.conv2(x)
        x = torch.tanh(x)
        x = self.conv3(x)
        v1 = torch.tanh(x)
        return v1
# Inputs to the model
input = torch.randn(10, 3, 249, 249)
