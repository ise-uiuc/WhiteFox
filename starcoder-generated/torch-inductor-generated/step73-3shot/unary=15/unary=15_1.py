
class Model(torch.nn.Module):
    def __init__(self, padding0):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=padding0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros')
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64) # padding=-4
