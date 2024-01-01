
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(1, 257, 1, stride=2, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        v1 = self.conv_1(x)
        v3 = self.tanh(v1)
        return v3
# Inputs to the model
x = torch.randn(1, 1, 299, 299)
