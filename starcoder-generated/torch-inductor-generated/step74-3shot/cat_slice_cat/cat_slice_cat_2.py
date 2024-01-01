
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4, x5):
        v1 = torch.cat([x1, x2, x3], dim=1)
        v2 = v1[:, 1000000000000000000]
        v3 = v2[:, :y]
        v4 = torch.cat((v1, v3), dim=1)
        v5 = torch.cat([v4, x4, x5])
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1000000000000000000)
x2 = torch.randn(1, 26)
x3 = torch.randn(1, 30)
x4 = torch.randn(1, 24)
x5 = torch.randn(1, 48)
