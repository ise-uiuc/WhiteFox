
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.addmm = torch.addmm
        self.cat = torch.cat
    def forward(self, x):
        x = self.addmm(x, torch.rand(2, 2), torch.rand(2,2))
        x = self.cat([x, x], dim=1)
        y = x + x
        return y
# Inputs to the model
x = torch.randn(2, 2)
