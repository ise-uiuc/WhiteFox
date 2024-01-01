
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x):
        v1 = torch.cat(x)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:v1.size(2)]
        v4 = torch.cat(x + [v2, v3])
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
x2 = torch.randn(1, 1, 28, 28)
x3 = torch.randn(1, 1, 28, 28)
