
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1_0, x1_1, x1_2):
        v1 = torch.cat([x1_0, x1_1, x1_2], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:9223372036854775807]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1_0 = torch.randn(1, 9223372036854775807, 2, 5)
x1_1 = torch.randn(1, 9223372036854775807, 6, 2)
x1_2 = torch.randn(1, 9223372036854775807, 3, 3)
