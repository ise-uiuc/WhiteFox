
class Model(torch.nn.Module):
    def __init__(self, dim=1):
        super().__init__()
 
    def forward(self, *xs):
        t1 = torch.cat(xs, dim=1)
        t2 = t1[:, 0:9223372036854775807]
        t3 = t2[:, 0:xs[0].size(dim)]
        t4 = torch.cat([t1, t3], dim=1)
        return t4

# Initializing the model
m = Model(dim=1)

# Inputs to the model
