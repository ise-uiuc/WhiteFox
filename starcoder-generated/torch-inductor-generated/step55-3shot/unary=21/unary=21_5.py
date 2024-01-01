
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_channels=64, out_channels=48, kernel_size=38, stride=1, padding=0, dilation=1, groups=1, bias=True)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 64, 1148)
