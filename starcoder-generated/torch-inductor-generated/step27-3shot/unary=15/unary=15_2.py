
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Conv2d(1, 128*2**i, 1, stride=1, padding=0) for i in range(7)])
    def forward(self, x1):
        v1 = []
        for i in range(7):
            v2 = self.layers[i](x1)
            v1.append(v2)
        return torch.cat(tuple(v1), 1)
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
