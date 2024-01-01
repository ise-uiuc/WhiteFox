
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, *args):
        args = torch.cat(args, dim=1)
        args_1 = args[:, 0:9223372036854775807]
        args_2 = args[:, 0:2]
        args = torch.cat([args, args_2], dim=1)
        return args

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 5, 64, 64)
