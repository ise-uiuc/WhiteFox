
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        x3 = torch.cat([x1, x2], dim=1)
        x4 = x3[:, 0:9223372036854775807]
        x5 = x4[:, 0:65535]
        x6 = torch.cat([x3, x5], dim=1)
        return x6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 128, 128)
