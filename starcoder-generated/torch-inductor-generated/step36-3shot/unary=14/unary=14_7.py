
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_0 = torch.nn.ConvTranspose2d(1593, 44, 1, stride=1, padding=0)
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(2528, 39128, 1, stride=1, padding=0)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(44, 2528, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose_0(x1)
        v2 = torch.sigmoid(v1)
        v4 = self.conv_transpose_1(x1)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        v7 = torch.cat([v2, v6], dim=1)
        v7 = torch.flatten(v7)
        v8 = self.conv_transpose_2(v7.reshape([7, 4364, 5, 5]))
        v9 = torch.sigmoid(v8)
        v10 = v8 * v9
        return v10
# Inputs to the model
x1 = torch.randn(1, 1593, 41, 41)
