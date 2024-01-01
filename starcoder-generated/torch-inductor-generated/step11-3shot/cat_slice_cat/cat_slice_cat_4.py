
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x):
        x1 = torch.transpose(x, 1, 2)
        x2 = torch.transpose(x, 1, 3)
        x3 = torch.cat([x1, x2], dim=1)
        x4 = x3[:, :, :, 0:9223372036854775807]
        x5 = x4[:, :, :, 0:494967295]
        x6 = torch.cat([x3, x5], dim=3)
        x7 = torch.transpose(x6, 1, 2)
        x8 = torch.transpose(x7, 1, 3)
        return x8

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(4, 64, 1, 1792)
