
class ModelTanh(torch.nn.Module):
    def __init__(self):
         super().__init__()
         self.conv1 = torch.nn.Conv2d(1, 128, (1, 1), stride=1, padding=0, dilation=1, groups=1, bias=False)
         self.conv2 = torch.nn.Conv2d(128, 128, (1, 1), stride=1, padding=0, dilation=1, groups=128, bias=False)
         self.conv3 = torch.nn.Conv2d(128, 128, (3, 3), stride=1, padding=1, dilation=1, groups=128, bias=False)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.tanh(v1)
        v3 = self.conv2(v2)
        v4 = torch.tanh(v3)
        v5 = self.conv3(v4)
        v6 = torch.tanh(v5)
        return v6
# Inputs to the model
x = torch.randn(1, 1, 32, 32)
