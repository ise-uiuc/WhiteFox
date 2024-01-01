
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, *x1):
        v1 = torch.cat(list(x1), dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:size]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x11 = torch.randn(1, 3, 256, 256)
x12 = torch.randn(1, 3, 480, 640)
