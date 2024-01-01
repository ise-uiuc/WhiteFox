
class Model(torch.nn.Module):
    def __init__(self, dilation_val):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, stride=1, dilation=dilation_val, padding=1)
    def forward(self, x):
        x = self.conv1(x)
        x = torch.tanh(x)
        x = self.conv2(x)
        x = torch.nn.functional.interpolate(x, mode='bilinear')
        return x
dilation_val = 1
# Inputs to the model
x = torch.randn(1, 3, 32, 32)
