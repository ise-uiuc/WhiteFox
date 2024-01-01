
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, inputs):
        v1 = torch.cat(inputs, dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:12]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10, 5, 6)
x2 = torch.randn(1, 12, 5, 6)
x3 = torch.randn(1, 14, 5, 6)
inputs = [x1, x2, x3]
