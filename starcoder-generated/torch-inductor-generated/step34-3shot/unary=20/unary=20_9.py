
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
    def forward(self, x1):
        v0 = x1.transpose(2, 1)
        v1 = self.conv_t(v0)
        v2 = v1.transpose(2, 1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 16, 24, 32)
