
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(48, 26, 7, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv1(v6)
        v8 = torch.flatten(v7, start_dim=1)
        v9 = self.conv1(v7)
        v10 = torch.flatten(v7, start_dim=-1)
        v11 = self.conv1(v7)
        v12 = torch.mean(v7, dim=[0, 2, 3])
        return (v8, v10, v12)
# Inputs to the model
x1 = torch.randn(14, 48, 27, 27)
