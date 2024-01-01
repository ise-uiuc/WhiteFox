
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = torch.nn.Linear(10, 10)
        self.l1 = torch.nn.Linear(10, 10)
 
    def forward(self, x1):
        x2 = self.l0(x1)
        x3 = self.l1(x1)
        x4 = x2 + 3
        x4 = torch.min(x4, torch.tensor(6, dtype=torch.float))
        x5 = x4 / 6
        x6 = x3.add(x5)
        return x6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10)
