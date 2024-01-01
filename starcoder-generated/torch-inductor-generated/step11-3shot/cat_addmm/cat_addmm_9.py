
class FC(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.randn((128, 25, 1, 1))
        self.bias = torch.randn(128)
 
    def forward(self, x):
        dim = 1
        t1 = torch.addmm(self.bias, x, self.weight.squeeze())
        t2 = torch.cat([t1], dim)
        return t2

# Initializing the model
m = FC()

# Inputs to the model
x = torch.randn(300, 128, 16)
