
class Model(torch.nn.Module):
    def forward(self, x1):
        v1 = x1.mean(dim=1)
        v2 = v1.clamp(min=0.0, max=6.0)
        v3 = v1 * v2
        v4 = v3 / float(6)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
