
class SubModel(nn.Module):
    def forward(self, x):
        t1 = torch.addmm(x, t, u)
        t2 = torch.cat((t1, t1, t1), dim=1)
        return t2
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.submodel = SubModel()
    def forward(self, x):
        return self.submodel(x)
# Inputs to the model
x = torch.randn(2, 2)
t = torch.rand(4, 2)
u = torch.rand(4, 2)
