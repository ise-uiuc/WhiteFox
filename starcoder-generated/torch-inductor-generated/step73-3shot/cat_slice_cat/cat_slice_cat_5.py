
class Model(torch.nn.Module):
    def forward(self, x1):
        size = 5
        v1 = torch.cat([x1, x1], dim=1)
        v2 = v1[:, 0:size]
        v3 = v2[:, 0:1]
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
