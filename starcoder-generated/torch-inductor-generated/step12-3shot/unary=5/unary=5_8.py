
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose1d(64, 64, 3, stride=1, padding=3)
        self.dropout = torch.nn.Dropout(p=0.2132573252511139)
        self.flatten = torch.nn.Flatten()
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.dropout(v6)
        v9 = self.flatten(v7)
        return v9
# Inputs to the model
x1 = torch.randn(1, 64, 6)
