
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose2d = torch.nn.ConvTranspose2d(3, 2, 2, stride=(1, 2), padding=1)
    def forward(self, input1):
        vconv_transpose2d = self.conv_transpose2d(input1)
        return vconv_transpose2d
# Inputs to the model
input1 = torch.randn(1, 3, 3, 3)
