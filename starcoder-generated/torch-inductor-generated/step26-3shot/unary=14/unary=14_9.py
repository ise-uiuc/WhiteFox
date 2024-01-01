
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(2048, 2048, 1, stride=1, padding=0)
        self.groupnorm_1 = torch.nn.GroupNorm(8, 2048, eps=1e-05, affine=True)
    def forward(self, x1):
        v1 = self.conv_transpose_3(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.groupnorm_1(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 2048, 10, 10)
