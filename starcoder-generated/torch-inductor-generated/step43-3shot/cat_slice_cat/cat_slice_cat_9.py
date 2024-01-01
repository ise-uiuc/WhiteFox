
def Model():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x1, x2, x3):
            t = torch.cat([x1, x2])
            t2 = t[:, 0:10]
            t3 = t2[:, 0: t.size()[1]]
            t4 = torch.cat([t, t3])
            return t4 + x3
    return Model()

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 3, 10)
x2 = torch.randn(3, 3, 10)
x3 = torch.randn(3, 3, 10)
