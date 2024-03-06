
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 6, 7, stride=1, padding=3)
        self.batch_norm_1d = torch.nn.BatchNorm3d(6)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.batch_norm_1d(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 120, 200)
x2 = torch.randn(1, 3, 60, 100)