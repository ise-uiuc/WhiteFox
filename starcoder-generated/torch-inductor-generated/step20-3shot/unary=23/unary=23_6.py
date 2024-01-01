
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convTranspose1d = torch.nn.ConvTranspose1d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.convTranspose1d(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 35)
