
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1267 = torch.nn.ConvTranspose2d(512, 512, 1, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose_1267(x1)
        v2 = torch.nn.ReLU()(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 512, 3, 3)
