
class Model(torch.nn.Module):
    def forward(self, x0, x1):
        v0 = torch.cat([x0, x1], dim=0)
        v1 = v0[:, :18446744073709551615]
        v2 = v1[:, :18446744073709551615]
        v3 = torch.cat([v0, v2], dim=1)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x0 = torch.randn(2, 1024)
x1 = torch.randn(2, 192)
