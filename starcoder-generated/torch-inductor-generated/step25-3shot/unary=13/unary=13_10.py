
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(192, 256)
        self.l2 = torch.nn.Linear(256, 256)
        self.l3 = torch.nn.Linear(256, 192)
 
    def forward(self, x1):
        v1 = self.l1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.l2(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        v7 = self.l3(v6)
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 192)
