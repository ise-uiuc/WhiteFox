
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(24, 79, kernel_size=(6, 6), stride=1, padding=2, groups=2)
        self.batch_norm = torch.nn.BatchNorm2d(79)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = self.batch_norm(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 24, 31, 63)
