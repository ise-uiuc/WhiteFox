
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3):
        v1 = torch.cat([x1, x2, x3], dim=1)
        v3 = v1[:, 0:6339]
        v4 = v3[:, 3968:6339]
        v5 = torch.cat([v1, v4], dim=1)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128, 256, 256)
x2 = torch.randn(1, 128, 256, 256)
x3 = torch.randn(1, 128, 256, 256)
