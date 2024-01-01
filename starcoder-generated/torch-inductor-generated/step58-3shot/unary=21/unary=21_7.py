
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(108, 108, (16, 16), stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=108, bias=True)
        self.conv2 = torch.nn.Conv2d(108, 1, (1, 1), stride=1, padding=0, dilation=1, groups=108, bias=True)
    def forward(self, x):
        h1 = self.conv1(x)
        h2 = torch.tanh(h1)
        return self.conv2(h2)
# Inputs to the model
x = torch.randn(58, 108, 17, 19)
