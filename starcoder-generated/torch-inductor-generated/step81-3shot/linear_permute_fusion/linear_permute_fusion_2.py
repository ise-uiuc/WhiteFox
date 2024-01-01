
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.block1 = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 1, padding=1, bias=False), torch.nn.LeakyReLU())
        self.block2 = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 1, bias=False), torch.nn.BatchNorm2d(1), torch.nn.LeakyReLU(), torch.nn.Softmax())
    def forward(self, x):
        v0 = x
        v1 = self.block1(v0)
        v2 = v1.permute(0, 2, 3, 1)
        v3 = self.block2(v2)
        v4 = v3.permute(0, 3, 1, 2)
        v5 = (v0 + v4) / 2.0
        return v5
# Inputs to the model
x = torch.randn(1, 1, 2, 2)
