
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3):
        t1 = torch.cat([x1, x2, x3], dim=1)
        t2 = t1[:, 0:9223372036854775807]
        t3 = t2[:, 0:x1.size()[1]]
        t4 = torch.cat([t1, t3], dim=1)
        return t4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.rand((1, 2, 3))
x2 = torch.rand((1, 2, 3))
x3 = torch.rand((1, 2, 3))

