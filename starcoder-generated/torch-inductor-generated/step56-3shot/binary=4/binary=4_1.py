
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(6, 1)
        self.linear2 = torch.nn.Linear(6, 1)
 
    def forward(self, x1):
        t1 = self.linear1(x1)
        t2 = self.linear2(x1)
        v6 = t1 + t2
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 6)
x2 = torch.randn(1, 6)
