
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v = []
        for var in range(2):
            for i in range(1):
                t1 = torch.mm(x[i], var)
                t2 = torch.mm(x[i], 1)
                t2 = torch.cat([t1, t2, t1, t2, t1, t2])
                v.append(t2)
        return torch.cat(v)
# Inputs to the model
x = torch.zeros([2, 2])
