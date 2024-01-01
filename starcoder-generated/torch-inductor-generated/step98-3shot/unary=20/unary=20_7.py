
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(255, 255, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
    def forward(self, x1):
        v2 = torch.sigmoid(self.conv_t(x1))
        return v2
# Inputs to the model
x1 = torch.randn(1, 255, 128, 128)
