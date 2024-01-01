
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 1, 1, stride=0, padding=0, dilation=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v10 = v1 * 0.2
        v11 = v1 * v1 * v1
        v12 = v11 * 6.8625
        v13 = v1 + v12
        v14 = v13 * 10.188842994226305
        v15 = torch.tanh(v14)
        v16 = v15 + 1
        v17 = v1 * 3.2766008787968
        v9 = v17 + v10 * v16
        return v9
# Inputs to the model
x1 = torch.randn(3, 3, 32, 32)
