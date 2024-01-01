
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_0 = torch.nn.ConvTranspose2d(8, 8, 3, stride=17, padding=1, bias=False)
        self.layer_1 = torch.nn.ConvTranspose2d(8, 8, 3, stride=17, padding=1)
    def forward(self, x1):
        v1 = self.layer_0(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.layer_1(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 8, 128, 128)
