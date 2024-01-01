
class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t1 = torch.addmm(torch.addmm(torch.addmm(x, x, x), x, x), x, x)
        t2 = torch.cat([t1], dim=2)
        return t2
# Inputs to the model
x = torch.randn(2, 2)
