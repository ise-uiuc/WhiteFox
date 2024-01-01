
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose3d(20, 34, kernel_size=(11, 11, 11), stride=(1, 1, 1), padding=(5, 5, 5), bias=False)
    def forward(self, x18):
        v1 = self.conv_t(x18)
        v2 = v1 > 0
        v3 = v1 * 0.040
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x18 = torch.randn(10, 20, 19, 27, 37)
