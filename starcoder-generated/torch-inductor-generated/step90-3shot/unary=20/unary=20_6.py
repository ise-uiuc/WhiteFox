
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose1d(52, 20, kernel_size=3, stride=1)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = torch.max(v1, 0)
        return v2[0]
# Inputs to the model
x1 = torch.randn(64, 52, 88832)
