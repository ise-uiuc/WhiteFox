
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        v1 = torch.cat([x1, x2], dim=3)
        v2 = v1[:, 0:1073741823]
        v3 = v2[:, 0:size]
        v4 = torch.cat([v1, v3], dim=3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 2, 64)
x2 = torch.randn(1, 3, 1, 32)
