
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 * 0.5
        v3 = torch.mul(v1, 0.7978845608028654)
        v4 = self.conv(x)
        v5 = v4 * 0.044715
        v6 = v4 + v5
        v7 = v3 + v6
        v8 = torch.tanh(v7)
        v9 = v2 * v8
        return v9

m = Model()
x = torch.randn(1, 3, 64, 64)
