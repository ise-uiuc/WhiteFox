
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(10, 2, 1, stride=2, padding=1)
        self.dropout = torch.nn.Dropout(p=0.8686074807437347)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.dropout(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 10, 32, 32)
