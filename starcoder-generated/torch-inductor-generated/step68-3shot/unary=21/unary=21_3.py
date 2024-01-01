
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
    def forward(self, x):
        x = self.conv1(x)
        x = torch.tanh(x)
        x = self.conv2(x)
        return x
# Inputs to the model
x = torch.rand(1, 1, 47, 63)
