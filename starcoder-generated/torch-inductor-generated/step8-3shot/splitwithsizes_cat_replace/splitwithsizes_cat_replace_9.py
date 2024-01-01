
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        t1 = torch.split(x1, (32, 64, 224))
        t2 = t1[0] * 0.5
        t3 = t2 + t1[1]
        t4 = torch.cat([t3, t2, t1[2]])
        t5 = t4 / 3.0
        t6 = t3 + 1
        return t6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64, 64)
