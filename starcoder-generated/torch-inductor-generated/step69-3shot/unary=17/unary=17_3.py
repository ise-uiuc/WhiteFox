
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose1d(32, 64, stride=4, groups=1)
        self.conv = torch.nn.Conv1d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, groups=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v1 = torch.tanh(v1)
        v2 = torch.nn.functional.gelu(v1)
        v3 = self.conv(v2)
        v3 = torch.nn.functional.sigmoid(v3)
        return v3
# Inputs to the model
x1 = torch.randn(1, 32, 40)
