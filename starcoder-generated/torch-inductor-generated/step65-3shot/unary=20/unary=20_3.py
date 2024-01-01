
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(11, 1, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0))
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 11, 84, 31)
