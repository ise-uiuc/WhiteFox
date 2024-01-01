
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(19, 5, 8, stride=2, padding=0)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(5, 8, 2, stride=(4, 2), padding=0)
    def forward(self, x):
        v1 = self.conv_transpose1(x)
        v2 = v1 + 0.03
        v3 = v2 * 0.4324885
        v4 = v2 - 0.7
        v5 = torch.sqrt(v4)
        v6 = v3 * v5
        v7 = v6 + 0.8
        v8 = v7 * -1.2
        v9 = self.conv_transpose2(v8)
        return v9
