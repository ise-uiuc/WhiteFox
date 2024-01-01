
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4):
        v1 = torch.cat((x1, x2, x3, x4), dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:67]
        v4 = torch.nn.functional.pixel_shuffle(v3, 32)
        v5 = v4[:, :, 0:16, 0:16]
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 196, 1, 1)
x2 = torch.randn(1, 134, 1, 1)
x3 = torch.randn(1, 98, 1, 1)
x4 = torch.randn(1, 67, 1, 1)
