
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_0 = torch.nn.ConvTranspose2d(512, 256, 1, 1, 0)
        self.conv_transpose_0.weight.data = torch.nn.init.xavier_normal_(self.conv_transpose_0.weight.data)
        self.conv_transpose_0.bias.data = torch.nn.init.xavier_normal_(self.conv_transpose_0.bias.data)
    def forward(self, x1):
        v1 = self.conv_transpose_0(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 512, 4, 8)
