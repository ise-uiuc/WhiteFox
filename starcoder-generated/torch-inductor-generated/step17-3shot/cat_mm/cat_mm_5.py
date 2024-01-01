
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([])
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        for _ in range(4):
            self.layers.append(v1)
        return torch.cat(self.layers, 1)
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 2)
