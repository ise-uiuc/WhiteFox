
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.s1 = torch.nn.Linear(256, 256)
        self.s2 = torch.nn.Linear(256, 9223372036854775807)
 
    def m1(self, x1):
        return self.s1(x1)
 
    def m2(self, x2):
        return self.s2(x2)
 
    def forward(self, x1):
        v1 = self.m1(x1)
        v2 = self.m2(v1)
        v3 = torch.cat([v1, v2], dim=1)
        v4 = v3[:, 0:size]
        v5 = torch.cat([v3, v4], dim=1)
        return v5
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 256)
x2 = torch.randn(1, 256)
