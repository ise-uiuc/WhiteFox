
class Model(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(n):
            self.convs.append(nn.Conv2d(4, 4, 3, stride=2, padding=1))        
    def forward(self, x):
        v1 = self.convs[0](x)
        v2 = self.convs[1](v1)
        v3 = v2 > 0
        v4 = v2 * -0.001
        v5 = torch.where(v3, v2, v4)
        v6 = self.conv[1](v5)
        return v6
n = 2
# Inputs to the model
x1 = torch.randn(1, 4, 448, 448)
