
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_tranpose_0 = torch.nn.ConvTranspose3d(480, 72, 3, padding=1)
    def forward(self, x1):
        v1 = self.conv_tranpose_0(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 480, 57, 60, 24)
