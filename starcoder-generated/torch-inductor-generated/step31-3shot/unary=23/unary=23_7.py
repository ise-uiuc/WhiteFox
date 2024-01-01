
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(64, 64, kernel_size=1, stride=1, padding=0, dilation=1, groups=8, bias=False)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.linear = torch.nn.Linear(1, 1)
        self.conv = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0, dilation=1, groups=1, bias=False)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.softmax(v1)
        v3 = self.linear(v2)
        v4 = self.conv(v2)
        v5 = v3 + v4
        v6 = torch.tanh(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 64, 224, 224)
