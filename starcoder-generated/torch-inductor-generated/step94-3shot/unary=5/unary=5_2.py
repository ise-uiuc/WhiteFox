
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=100, out_features=200, bias=False)
        self.conv_transpose = torch.nn.ConvTranspose2d(200, 200, 1, padding=0)
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv_transpose(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 100)
