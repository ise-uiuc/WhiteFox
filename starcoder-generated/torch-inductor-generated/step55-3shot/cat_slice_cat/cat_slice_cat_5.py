
class Model(torch.nn.Module):
    def __init__(self, size, dim):
        super().__init__()
        self.size = size
        self.dim = dim
 
    def forward(self, xs):
        ts1 = [x[:, :, 0:self.size, 0:self.size] for x in xs]
        ts2 = [x[:, 0, 0:self.size, 0:self.size] for x in ts1]
        ts3 = [x[:, 0, 0:self.size, 0:self.size] for x in ts2]
        ts4 = [torch.cat([x, y], dim=1) for x, y in zip(ts1, ts3)]
        return ts4

# Initializing the model
m = Model(32, 2)

# Inputs to the model
xs = [torch.randn(1, 1, 64, 64)]
