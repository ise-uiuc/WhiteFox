
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, weight = torch.randn((45, 10)), bias = torch.randn((10, ))):
        v1 = torch.matmul(x1, weight)
        return v1 - bias

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(128, 45)
