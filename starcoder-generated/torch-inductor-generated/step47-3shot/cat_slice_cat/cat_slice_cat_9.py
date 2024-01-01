
class Model(torch.nn.Module):
    def __init__(self):

    def forward(self, x1, x2):
        v1 = torch.cat([x1, x2], 1)
        v2 = v1[:, ::256]
        v3 = v2[:, 127::2]
        v4 = torch.cat([v1, v3], 1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1024, 25, 25)
x2 = torch.randn(1, 1024, 25, 25)
