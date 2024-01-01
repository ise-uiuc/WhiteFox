
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(100, 128)
        self.linear2 = torch.nn.Linear(256, 128)
        self.linear3 = torch.nn.Linear(6144, 128)
        self.linear4 = torch.nn.Linear(100, 128)
 
    def forward(self, x1, x2, x3, x4):
        v1 = self.linear1(x1)
        v2 = self.linear2(x2)
        v3 = self.linear3(x3)
        v4 = self.linear4(x4)
        v5 = torch.addmm(v1, v2, v3)
        v6 = torch.cat([v1, v2, v3, v4], 1)
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 768)
x2 = torch.randn(1, 3072)
x3 = torch.randn(1, 8192)
x4 = torch.randn(1, 50)
