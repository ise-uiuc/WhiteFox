
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3, 8, 3, stride=2, padding=1, dilation=2)
    def forward(self, x):
        x1 = self.conv2d(x)
        x2 = torch.nn.functional.relu6(x1)
        x3 = self.conv2d(x2)
        x4 = torch.nn.functional.tanh(x3)
        return x4
# Inputs to the model
x = torch.randn(1, 3, 256, 256)
