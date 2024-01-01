
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x):
        v1 = torch.addmm(x, self.mm1, self.mm2)
        v2 = torch.cat([v1, v1], dim=0)
        return v2
 
m = Model()

# Generating matrix variables
v1 = torch.randn(10)
