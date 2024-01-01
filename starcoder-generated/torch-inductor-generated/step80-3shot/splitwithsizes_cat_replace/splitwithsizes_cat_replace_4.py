
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.ModuleList([torch.nn.Linear(i, 3) for i in range(8)])
    def forward(self, v1):
        _0 = torch.split(v1, [6, 6], dim=1)
        _1 = torch.cat([_0[0], _0[1]], dim=1)
        return (_1, torch.split(v1, [6, 2], dim=1))
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
