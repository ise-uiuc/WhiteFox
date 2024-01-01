
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.constant1 = torch.empty(1)
        self.constant2 = torch.empty(1)
        self.constant3 = torch.empty(1)
 
    def forward(self, x1):
        t1 = torch.cat([x1, self.constant1, self.constant2], dim=1)
        t2 = t1[:, 1:3]
        t3 = torch.cat([t1, self.constant3], dim=1)
        return torch.cat([t1, t2, t3], dim=1)

# Initializing the model
model = Model()

# Inputs to the model
x1 = torch.randn(1, 2, 64, 64)
