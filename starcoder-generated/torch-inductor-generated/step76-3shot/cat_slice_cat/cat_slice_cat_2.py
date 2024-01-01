
class Model(torch.nn.Module):
    def forward(a, b, c, d):
        t1 = torch.cat([a, b], dim=1)
        t2 = t1[:, 0:-1]
        t3 = t2[:, 0:-1]
        t4 = torch.cat([t1, t3], dim=1)
        return t4

# Initializing the model
m = Model()

