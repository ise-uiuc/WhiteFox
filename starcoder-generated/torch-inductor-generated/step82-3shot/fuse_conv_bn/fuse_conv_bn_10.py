
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(2):
            self.layers.append(torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, bias=True), torch.nn.BatchNorm2d(4)))
    def forward(self, x):
        x1 = self.layers[0](x)
        x2 = self.layers[1](x)
        return x1 + x2
# Inputs to the model
x = torch.randn(1, 3, 10, 10)
