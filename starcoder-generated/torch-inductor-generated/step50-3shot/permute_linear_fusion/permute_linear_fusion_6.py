
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.rand_like(x1)
        x2 = x1.unsqueeze(dim=1)
        v2 = torch.rand_like(x1)
        x3, i3 = torch.max(x2, dim=1)
        v3 = torch.index_select(v1, dim=1, index=torch.remainder(i3, 1))
        v3 = v3.unsqueeze(dim=2)
        x4, i4 = torch.max(v1, dim=1)
        v4 = torch.index_select(x1, dim=1, index=torch.remainder(i4, 2))
        v4 = v4.unsqueeze(dim=1)
        v5 = torch.rand_like(i3)
        x5 = torch.index_select(x4, dim=1, index=torch.remainder(i3, 3))
        x5 = x5 + v4
        x5 = torch.sigmoid(x5)
        x6, i6 = torch.max(v5, dim=1)
        v6 = torch.index_select(v1, dim=1, index=torch.remainder(i6, 1))
        x5 = x6.unsqueeze(dim=2)
        v6 = torch.rand_like(v1)
        v6 = torch.index_select(v6, dim=1, index=torch.remainder(i6, 3))
        x6 = x6 + v6
        x7 = torch.sigmoid(x6)
        v7 = v5.unsqueeze(dim=1)
        x7 = v7 * x7
        v7 = x7.permute(0, 2, 1)
        v7 = torch.rand_like(v7)
        x8 = x1 + v7
        v8 = x8.permute(0, 2, 1)
        v8 = x8.unsqueeze(dim=1)
        x9 = torch.sigmoid(x8)
        x10 = torch.cat([x3, x5], dim=1)
        v10 = v1 * x10
        x11 = x9 * v10
        x11 = x11.permute(0, 2, 1)
        v11 = torch.rand_like(v1)
        v11 = x11.permute(0, 2, 1)
        x12 = x10 + v11
        return x10, x11, x12
# Inputs to the model
x1 = torch.randn(1, 2, 4)
