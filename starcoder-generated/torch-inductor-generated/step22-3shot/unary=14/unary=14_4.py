
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dconv_3_1 = torch.nn.ConvTranspose2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), output_padding=(2, 2))
    def forward(self, x1):
        v1 = self.dconv_3_1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 512, 8, 8)
