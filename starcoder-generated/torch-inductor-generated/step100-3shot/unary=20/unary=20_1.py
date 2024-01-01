
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(16, 37, kernel_size=(1, 86), stride=(1, 86), padding=(0, 85))
        self.sigm = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = self.sigm(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 16, 3, 22)
