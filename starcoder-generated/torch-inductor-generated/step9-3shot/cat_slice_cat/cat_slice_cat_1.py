
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3):
        x = torch.cat([x1, x2, x3], dim=1)
        y = x[:, 0:x.shape[1]]
        z = x[:, x.shape[1]:]
        z = torch.cat([x, z], dim=1)
        return z

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
x3 = torch.randn(1, 3, 64, 64)
