
class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t1 = torch.addmm(x, torch.randn(2, 2), torch.randn(2, 2))
        t2 = torch.rand(2)
        t3 = torch.stack((t2, t2 * 2), dim=0)
        t4 = torch.cat((t1, t3), dim=0)
        return t4
# Inputs to the model
x = torch.randn(2, 2)
