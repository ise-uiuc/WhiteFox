
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        v3 = torch.cat([x2.reshape([2, 4, 8])], dim=1)
        v4 = v1[:, 0:3]
        v5 = v2[:, 0:size]
        v6 = torch.cat([v1, v3], dim=1)
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 8, 8)
x2 = torch.randn(2, 128, 8)
