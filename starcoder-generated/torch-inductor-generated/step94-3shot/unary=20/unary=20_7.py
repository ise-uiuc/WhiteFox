
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(96, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_t_0 = torch.nn.ConvTranspose2d(64, 32, kernel_size=(1, 1), stride=(2, 2), padding=(0, 0))
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv_t_0(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 96, 224, 224)
