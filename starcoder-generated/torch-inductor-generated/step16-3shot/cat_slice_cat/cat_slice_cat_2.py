
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        v1 = torch.cat([x1, x2], dim=1)
        v2 = v1[:, -1]
        v3 = v1[:, :v2]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 4096, 1)
x2 = torch.randn(1, 3, 1, 1)
_output_ = m(x1, x2)

