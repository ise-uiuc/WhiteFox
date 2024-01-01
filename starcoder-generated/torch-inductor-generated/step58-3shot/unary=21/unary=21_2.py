
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(428, 428, (3, 13), stride=(1, 2), padding=(1, 2), dilation=(1, 1), groups=3, bias=True)
        self.conv2 = torch.nn.Conv2d(428, 428, (1, 9), stride=1, padding=0, dilation=1, groups=1, bias=True)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.tanh(v1)
        v3 = self.conv2(v2)
        return torch.tanh(v3)
Inputs to the model
x = torch.randn(23, 428, 145, 87)
