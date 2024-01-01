
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = torch.nn.Linear(5, 5)
        self.lin2 = torch.nn.Linear(5, 5)
        self.lin3 = torch.nn.Linear(5, 5)
        self.lin4 = torch.nn.Linear(5, 5)
 
    def forward(self, x1):
        v1 = self.lin1(x1)
        v2 = v1 > 0
        v3 = self.lin2(v1) * 0.01
        v4 = torch.where(v2, v1, v3)
        return self.lin3(v4) + self.lin4(x1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5)
