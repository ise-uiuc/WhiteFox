
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3):
        v1 = torch.cat([x1, x2, x3], dim=1)
        v2 = v1[:, 0:]
        v3 = v2[:, 0:]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Inputs to the model
x1 = torch.randn(1, 2, 3)
x2 = torch.randn(1, 3, 7)
x3 = torch.randn(1, 6, 5)
