
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.b = nn.BatchNorm3d(3)
        self.a = nn.Conv3d(3, 3, 1)
    def forward(self, x):
        a = self.a(self.b(x))
        b = self.b(x)
        return a + b
