
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        self.t1 = torch.cat([x1, x2], dim=1)
        self.t2 = self.t1[:, 0:-1]
        self.t3 = self.t2[:, 0:-3]
        self.t4 = torch.cat([self.t1, self.t3], dim=1)
        return self.t4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 256, 256)
