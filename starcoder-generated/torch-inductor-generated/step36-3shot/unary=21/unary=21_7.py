
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(5, 2, 2, stride=2, bias=False) # output[4, 1, 4, 4]
        self.conv2 = torch.nn.Conv2d(2, 10, 5, padding=5, dilation=1, groups=1, bias=False)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.tanh(v2)
        return v3
# Inputs to the model
x = torch.randn(20, 5, 20, 30)
