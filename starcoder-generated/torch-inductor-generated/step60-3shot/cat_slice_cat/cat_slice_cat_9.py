
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, *args):
        args = list(args)
        v1 = torch.cat(args, dim=1)
        size = args[0].size(1) - 1
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:size]
        args.append(v3)
        v4 = torch.cat(args, dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
