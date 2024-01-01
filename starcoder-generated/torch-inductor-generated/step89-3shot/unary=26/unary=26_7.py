
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose3d(211, 314, 4, stride=(1, 3, 2), padding=(2, 3, 2), bias=True)
    def forward(self, x16):
        r1 = self.conv_t(x16)
        r2 = -0.242 * torch.tensor((-0.67, -0.01, -1.0, -0.21, -0.6, 1.84, -0.88, -0.35, 0.94, -0.15, -2.0, 1.74, 1.33, -0.81, 0.47, -0.03, 0.1, 0.98, 1.86, 0.46, 1.58, -0.8, 0.64, 0.01, 1.05, 2.43, 1.22, 0.98), dtype=torch.float32) * r1
        r3 = r2 > 0
        r4 = torch.where(r3, r1, r2)
        return -0.65 * r4
# Inputs to the model
x16 = torch.randn(8, 211, 22, 1, 1)
