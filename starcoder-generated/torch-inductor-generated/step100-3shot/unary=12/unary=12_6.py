
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_trans = torch.nn.ConvTranspose2d(1, 3, 8)
    def forward(self, x1):
        v1 = self.conv_trans(x1)
        return v1
# Inputs to the model
x1 = torch.randn(2, 1, 16, 16)
