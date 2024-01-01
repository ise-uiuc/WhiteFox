
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        t1 = torch.cat([x1, x1 * 2], dim=1)
        t2 = t1[:, 0:1024*1]
        t3 = t2[:, 0:99999999999]
        return torch.cat([t1, t3], dim=1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1023, 10)
