
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.split = torch.nn.Split(3, [4, 7], 1)
 
    def forward(self, x1):
        t1 = self.split(x1)
        t2 = [t1[0], t1[0], t1[0], t1[0], t1[1], t1[1], t1[1]]
        b = torch.cat(t2, self.split.dim)
        v0 = b + x1
        return v0

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
