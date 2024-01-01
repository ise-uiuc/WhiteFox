
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        m1 = torch.nn.BatchNorm2d(2)
        m2 = torch.nn.Conv2d(1, 2, 3)
        m3 = torch.nn.Dropout(0.1)
        m4 = torch.nn.Identity()
        self.register_modules(modules=dict(m1=m1, m2=m2, m3=m3, m4=m4))
    def forward(self, x):
        x = self.m1(x)
        x1 = self.m2(x)
        x2 = self.m3(x1)
        x3 = self.m4(x2)
        x4 = self.m4(x2)
        return x4, x3
# Inputs to the model
x = torch.randn(1, 1, 8, 8)
