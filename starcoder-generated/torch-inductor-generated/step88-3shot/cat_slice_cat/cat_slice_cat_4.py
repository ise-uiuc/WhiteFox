
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, *tensors):
        x1 = torch.cat(tensors, dim=1)
        x2 = x1[:, 0:9223372036854775807]
        x3 = x2[:, 0:x2.shape[2]//2]
        x4 = torch.cat([x1, x3], dim=1)
        return x4

# Initializing the model
m = Model()

# Inputs to the model
a, b, c, d = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 32, 64), torch.randn(1, 3, 32, 32), x = torch.randn(1, 3, 24, 48)
tensors = a, b, c, d, x
