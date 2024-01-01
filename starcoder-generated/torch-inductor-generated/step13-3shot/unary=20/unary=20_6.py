
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 1, kernel_size=21, stride=(1, 1), padding=(4, 23))
    def forward(self, v):
        v0 = v
        v1 = self.conv_t(v0)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
v = torch.randn(1, 2, 24, 5)
