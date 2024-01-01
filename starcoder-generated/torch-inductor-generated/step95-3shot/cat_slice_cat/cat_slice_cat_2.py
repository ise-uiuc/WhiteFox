
class Model(torch.nn.Module):
    def forward(self, x0, x1, x2):
        v0 = torch.cat([x0, x1, x2], -2)
        v1 = v0[:, :, :224]*0.5
        v2 = v0[:, :, 224:]*0.114
        v3 = (v1 - 0.5) + (v2 + 1.14)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x0 = torch.randn(1, 3, 256, 256)
x1 = torch.rand(1, 1, 224, 224)
x2 = torch.rand(1, 150, 80, 80)
