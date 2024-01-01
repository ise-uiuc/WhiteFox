
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x0):
        v1 = torch.cat(x0, dim=1)
        v2 = v1[:, 0:9223372036854775807]
        return v2, v1

# Initializing the model
m = Model()

# Inputs to the model
x0 = [
    torch.randn(1, 16, 224, 224),
    torch.randn(1, 16, 112, 112),
    torch.randn(1, 16, 56, 56)
]
