
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(153, 153, (20, 20), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=153, bias=True)
        self.conv2 = torch.nn.Conv2d(153, 153, (1, 1), stride=1, padding=0, dilation=1, groups=153, bias=True)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.tanh(v1)
        v3 = self.conv2(v2)
        v4 = torch.tanh(v3)
        return v4
# Inputs to the model
x = torch.randn(17, 153, 28, 28)
