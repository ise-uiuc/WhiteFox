
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pad = torch.nn.ZeroPad1d((0, 0, 1, 2))
        self.conv_0 = torch.nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1, bias=False)
        self.tanh_0 = torch.nn.Tanh()
    def forward(self, x0):
        x1 = self.pad(x0)
        x2 = self.conv_0(x1)
        x3 = self.tanh_0(x2)
        return x3
# Inputs to the model
x0 = torch.randn(1, 256, 2)
