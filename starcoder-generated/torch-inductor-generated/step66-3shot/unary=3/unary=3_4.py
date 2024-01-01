
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 35, 1, stride=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * -0.4510878435641825
        v3 = v1 * -0.09122562836694702
        v4 = torch.erf(v3)
        v5 = v4 + 0.37870897700509405
        v6 = v2 * v5
        v7 = v1 * 0.2767710722767271
        v8 = v1 * 0.20196397432171374
        v9 = torch.erf(v8)
        v10 = v9 + 1.0151378214411041
        v11 = v7 * v10
        v12 = v1 * -0.4151510464222559
        v13 = torch.erf(v8)
        v14 = v13 + 1.5500693513726679
        v15 = v12 * v14
        return v11 + v15
# Inputs to the model
x1 = torch.randn(1, 2, 6, 6)
