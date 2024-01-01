
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4):
        v1 = torch.cat([x1, x2, x3], dim=1)
        v2 = v1[:, 0:x2.size(1)]
        v3 = v2[:, 0:x4.size(3)]
        v4 = [v1, v3]
        v5 = torch.cat([v4, x4], dim=1)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4, 32, 32)
x2 = torch.randn(1, x1.size(1) - 1, x1.size(2), x1.size(3))
x3 = torch.randn(1, x2.size(1) - 1, x2.size(2), x2.size(3))
x4 = torch.randn(1, x3.size(1) + 2, x3.size(2), x3.size(3))
