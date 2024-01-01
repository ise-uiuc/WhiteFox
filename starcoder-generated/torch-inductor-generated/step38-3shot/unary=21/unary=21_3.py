
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 256, kernel_size=1)
        self.conv_1 = torch.nn.Conv2d(256, 1024, 1)
        self.conv_2 = torch.nn.Conv2d(1024, 5, 1)
        self.sig = torch.nn.Sigmoid()
    def forward(self, x):
        v1 = self.conv_1(self.conv(x))
        v2 = torch.tanh(v1)
        self.conv_2(v2)
        v3 = torch.tanh(v2)
        v4 = self.sig(self.conv_2(v3)).squeeze(1)
        return v4
# Inputs to the model
x = torch.randn(128, 16, 256, 256)
