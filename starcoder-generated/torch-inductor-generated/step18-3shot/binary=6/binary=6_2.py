
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1,1,1)
        self.linear2 = torch.nn.Linear(1,1,1)
 
    def forward(self, x1):
        z0 = x1.sum()
        z1 = z0 + x1.max()
        z2 = z1 / x1.min()
        z3 = torch.arange(x1.numel())
        z4 = z3[::2].float()
        z5 = z3[1:x1.numel()//2].sum()
        z6 = z3[x1.numel()//2:x1.numel()].float()
        y0 = z2/z5
        z7 = y0[z4].sum()
        z8 = z7 * z6
        y1 = x1 - z8
        return y1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(100)
