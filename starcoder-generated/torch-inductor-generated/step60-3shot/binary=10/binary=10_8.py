
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
        v1 = x1.flatten(1)
        v4 = torch.cos(v1)
        v2 = torch.erfinv(v4)
        v3 = torch.reshape(v2, shape=(-1, 49, 768))
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 49, 768)
