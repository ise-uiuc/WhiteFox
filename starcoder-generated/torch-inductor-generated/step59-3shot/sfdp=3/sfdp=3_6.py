
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = torch.nn.MatrixMult()
        self.m2 = torch.nn.Softmax(dim=-1)
        self.m3 = torch.nn.Dropout(p=0.5)
 
    def forward(self, x1, x2):
        v1 = self.m1(x1, x2)
        v2 = v1 * 0.5
        v3 = self.m2(v2)
        v4 = self.m3(v3)
        v5 = self.m3(v4)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32, 1024, 1024)
x2 = torch.randn(1, 32, 1024, 1024)
