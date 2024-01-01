
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x11, x12, x13):
        v7 = torch.cat([x11, x12, x13], dim=1)
        v8 = v7[:, 0:9223372036854775807]
        v9 = v8[:, -(v7.size(1) - 1):]
        v10 = torch.cat([v7, v9], dim=1)
        return v10

# Initializing the model
m = Model()

# Inputs to the model
x11 = torch.randn(1, 48, 1, 1)
x12 = torch.randn(1, 120, 1, 1)
x13 = torch.randn(1, 224, 1, 1)
