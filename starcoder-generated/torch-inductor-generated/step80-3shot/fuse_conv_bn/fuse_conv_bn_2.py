
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = []
        for i in range(3):
            torch.manual_seed(i)
            self.layers.append(torch.nn.Sequential(torch.nn.Conv2d(3, 3, 1), torch.nn.LeakyReLU(), torch.nn.BatchNorm2d(3, affine=False)))
    def forward(self, x1):
        for i in range(3):
            x1 = self.layers[i](x1)
        return x1
# Inputs to the model
x1 = torch.randn(1, 3, 6, 6)
