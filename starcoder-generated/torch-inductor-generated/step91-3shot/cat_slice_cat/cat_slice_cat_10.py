
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, __input__1, __input__2):
        v1 = torch.cat([__input__1, __input__2], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:__input__1.size(1)]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 3, 2)
x2 = torch.randn(1, 3, 3, 3)
