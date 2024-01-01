
class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = ModuleList([Module()])
    def forward(self, x):
        x1 = torch.split(x, 2, dim=0)[0]
        x2 = torch.split(x, 2, dim=0)[1]
        y = torch.cat((x1, x2))
        return y
# Inputs to the model
x1 = torch.randn(4, 2)
