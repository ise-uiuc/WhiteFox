
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 406, (71, 3), stride=(8, 3), padding=(12, 1), dilation=1, groups=1, bias=True)
        self.conv2 = torch.nn.Conv2d(406, 503, (1, 1), stride=1, padding=0, dilation=1, groups=1, bias=True)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.tanh(v1)
        v3 = self.conv2(v2)
        v4 = torch.tanh(v3)
        return v4
# Inputs to the model
x = torch.randn(4, 1, 100, 200)
