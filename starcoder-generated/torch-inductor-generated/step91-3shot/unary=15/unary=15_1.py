
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prelu1 = torch.nn.PReLU()
        self.prelu2 = torch.nn.PReLU()
        self.prelu3 = torch.nn.PReLU()
        self.prelu4 = torch.nn.PReLU()
        self.prelu5 = torch.nn.PReLU()
        self.prelu6 = torch.nn.PReLU()
        self.prelu7 = torch.nn.PReLU()
    def forward(self, x1):
        v1 = self.prelu1(x1)
        v2 = self.prelu2(v1)
        v3 = self.prelu3(v2)
        v4 = self.prelu4(v3)
        v5 = self.prelu5(v4)
        v6 = self.prelu6(v5)
        v7 = self.prelu7(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
