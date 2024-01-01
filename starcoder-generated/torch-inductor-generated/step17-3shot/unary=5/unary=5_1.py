
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(5, 12, kernel_size=4, stride=2, padding=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(12, 6, kernel_size=3, stride=3, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = v1 * 0.2689414213699951
        v3 = v1 * 0.5873179595718384
        v4 = v3 * 0.7071067811865476
        v5 = v4 * 0.5590169943749475
        v6 = (-1.5707963267948966 == v3)
        v7 = v2 * v5
        v8 = torch.tanh(v7)
        v9 = v8 + 1.1240370639648438
        v10 = v1 - v2
        v11 = torch.erf(v10)
        v12 = v11 + 0.7071067811865476
        v13 = v9 * v12
        return v13
# Inputs to the model
x1 = torch.randn(1, 5, 3, 6)
